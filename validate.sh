#!/bin/bash
# ==============================================================================
# validate.sh - SageMaker random forest demo: deployment check
# ------------------------------------------------------------------------------
#   - Notebook is InService and its project files are staged in S3.
#   - MCP connector: OAuth discovery, 401 without a token, then tools/list and
#     tools/call with a real Cognito token for a throwaway user (created here,
#     deleted on exit).
#   - One line per check, then a short summary.
#
# Starts nothing billable. If the endpoint is deployed, tools/call sends it
# one prediction.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

fail() { echo "FAILED"; echo "ERROR: $*" >&2; exit 1; }
step() { printf '%-58s' "$1..."; }
ok()   { echo "${1:-ok}"; }

OUT1="$(terraform -chdir=01-infrastructure output -json 2>/dev/null || echo '{}')"
OUT2="$(terraform -chdir=02-mcp output -json 2>/dev/null || echo '{}')"
v1() { jq -r ".$1.value // empty" <<<"${OUT1}"; }
v2() { jq -r ".$1.value // empty" <<<"${OUT2}"; }

REGION="$(v1 region)"; BUCKET="$(v1 bucket)"; NOTEBOOK="$(v1 notebook_name)"
MCP_URL="$(v2 mcp_endpoint)"; API_BASE="$(v2 api_base_url)"
POOL="$(v2 cognito_user_pool_id)"; CLIENT_ID="$(v2 mcp_client_id)"
CLIENT_SECRET="$(v2 mcp_client_secret)"; ROUTER="$(v2 router_function)"
[ -n "${BUCKET}" ] && [ -n "${MCP_URL}" ] || fail "Terraform outputs missing. Run ./apply.sh first."
export AWS_DEFAULT_REGION="${REGION}"

# ------------------------------------------------------------------------------
# Notebook
# ------------------------------------------------------------------------------
step "Notebook ${NOTEBOOK}"
STATUS="$(aws sagemaker describe-notebook-instance --notebook-instance-name "${NOTEBOOK}" \
  --query NotebookInstanceStatus --output text)"
[ "${STATUS}" = "InService" ] || fail "Notebook ${NOTEBOOK} is ${STATUS}. Lifecycle log: CloudWatch /aws/sagemaker/NotebookInstances, stream ${NOTEBOOK}/LifecycleConfigOnStart"

FILES="$(aws s3api list-objects-v2 --bucket "${BUCKET}" --prefix notebook-source/ \
  --no-paginate --query KeyCount --output text)"
[ "${FILES}" -ge 5 ] 2>/dev/null || fail "Project files missing under s3://${BUCKET}/notebook-source/ (found ${FILES})."
ok "InService"

# ------------------------------------------------------------------------------
# MCP connector
# ------------------------------------------------------------------------------
step "MCP OAuth discovery and 401 without a token"
REG="$(curl -sf --max-time 15 "${API_BASE}/.well-known/oauth-authorization-server" | jq -r .registration_endpoint)"
[ "${REG}" = "${API_BASE}/oauth/register" ] || fail "OAuth metadata did not advertise ${API_BASE}/oauth/register."

CODE="$(curl -s --max-time 15 -o /dev/null -w '%{http_code}' -X POST "${MCP_URL}" \
  -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}')"
[ "${CODE}" = "401" ] || fail "${MCP_URL} answered ${CODE} without a token; expected 401."
ok

# Throwaway user signed in through Cognito's admin API (IAM-authorized).
step "MCP tools/list with a Cognito token"
USER="mcp-validate-$(date +%s)@example.com"
PASS="Val-$(openssl rand -hex 8)-Ab1"
trap 'aws cognito-idp admin-delete-user --user-pool-id "${POOL}" --username "${USER}" >/dev/null 2>&1 || true' EXIT
aws cognito-idp admin-create-user --user-pool-id "${POOL}" --username "${USER}" \
  --message-action SUPPRESS \
  --user-attributes Name=email,Value="${USER}" Name=email_verified,Value=true > /dev/null
aws cognito-idp admin-set-user-password --user-pool-id "${POOL}" --username "${USER}" \
  --password "${PASS}" --permanent
HASH="$(printf '%s' "${USER}${CLIENT_ID}" | openssl dgst -sha256 -hmac "${CLIENT_SECRET}" -binary | base64)"
TOKEN="$(aws cognito-idp admin-initiate-auth --user-pool-id "${POOL}" --client-id "${CLIENT_ID}" \
  --auth-flow ADMIN_USER_PASSWORD_AUTH \
  --auth-parameters USERNAME="${USER}",PASSWORD="${PASS}",SECRET_HASH="${HASH}" \
  --query AuthenticationResult.AccessToken --output text)"

mcp() {
  curl -s --max-time 40 -X POST "${MCP_URL}" -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${TOKEN}" -d "$1"
}
TOOL="$(mcp '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | jq -r '.result.tools[0].name // empty')"
[ "${TOOL}" = "predict_equipment_failure" ] || fail "MCP tools/list failed with a valid token. Logs: /aws/lambda/${ROUTER}"
ok

# A cold serverless endpoint takes ~20s to answer the first call.
step "MCP tools/call (up to 30s if the endpoint is cold)"

CALL="$(mcp '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"predict_equipment_failure","arguments":{"readings":[{"temperature":81,"vibration":5.1,"operating_hours":11000}]}}}')"
TEXT="$(jq -r '.result.content[0].text // empty' <<<"${CALL}")"
if [ "$(jq -r '.result.isError // false' <<<"${CALL}")" = "false" ] && [ -n "${TEXT}" ]; then
  PREDICTION="$(head -1 <<<"${TEXT}" | sed 's/^1\. //')"
  ok
elif [[ "${TEXT}" == *"not deployed"* ]]; then
  PREDICTION="endpoint not deployed yet (python demo.py deploy)"
  ok "no endpoint yet"
else
  fail "MCP tools/call: ${TEXT:-${CALL}}"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
cat <<EOF

SageMaker Random Forest Demo
============================
JupyterLab : https://${NOTEBOOK}.notebook.${REGION}.sagemaker.aws/lab
             (sign in to the AWS console first)
MCP URL    : ${MCP_URL}
             (claude.ai: Settings > Connectors > Add custom connector; sign up on first connect)
MCP check  : ${PREDICTION}
Bucket     : s3://${BUCKET}

Notebook costs ~\$0.05/hour while running: python demo.py notebook-stop
EOF
