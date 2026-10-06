#!/bin/bash
# ==============================================================================
# add_mcp_user.sh - create a login for the MCP connector
# ==============================================================================
# Usage: ./add_mcp_user.sh you@example.com
#
# The connector's Cognito pool has self sign-up turned off, so users are
# created here, with your AWS credentials. Prints a generated password; the
# user signs in with it at the Cognito page Claude opens. No email is sent.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

EMAIL="${1:-}"
if [[ -z "${EMAIL}" || "${EMAIL}" != *@* ]]; then
  echo "Usage: ./add_mcp_user.sh you@example.com"
  exit 1
fi

OUTPUTS="$(terraform -chdir=02-mcp output -json 2>/dev/null || echo '{}')"
POOL="$(jq -r '.cognito_user_pool_id.value // empty' <<<"${OUTPUTS}")"
REGION="$(jq -r '.region // empty' 02-mcp/deployment.tfvars.json 2>/dev/null || true)"
if [[ -z "${POOL}" || -z "${REGION}" ]]; then
  echo "ERROR: The MCP connector is not deployed. Run ./apply.sh first."
  exit 1
fi

# Meets the pool policy: 12+ characters with upper, lower and a digit.
PASSWORD="Mcp-$(openssl rand -hex 6)-Ab1"

aws cognito-idp admin-create-user --region "${REGION}" --user-pool-id "${POOL}" \
  --username "${EMAIL}" --message-action SUPPRESS \
  --user-attributes Name=email,Value="${EMAIL}" Name=email_verified,Value=true > /dev/null
aws cognito-idp admin-set-user-password --region "${REGION}" --user-pool-id "${POOL}" \
  --username "${EMAIL}" --password "${PASSWORD}" --permanent

echo "NOTE: Created MCP user ${EMAIL}"
echo "NOTE: Password: ${PASSWORD}"
echo "NOTE: Remove later with: aws cognito-idp admin-delete-user --region ${REGION} --user-pool-id ${POOL} --username ${EMAIL}"
