#!/bin/bash
# ==============================================================================
# destroy.sh - Tear down the SageMaker random forest demo
# ==============================================================================
# Torn down in this order:
#
#   1. The MCP connector (terraform destroy in 02-mcp)
#        so nothing can call the model while the rest is removed
#   2. Python-created resources (demo.py cleanup --purge-data)
#        stop any running training job, delete the endpoint (and wait for it),
#        endpoint configs, models, their logs, and the demo's own S3 prefixes
#   3. Terraform-owned resources (terraform destroy in 01-infrastructure)
#        notebook instance (stopped, then deleted), IAM roles, bucket
#
# Step 2 must come before step 3: Terraform has never heard of the endpoint,
# and would delete the role it runs as while leaving it behind.
#
# The bucket is never force-emptied. If it holds objects this demo did not
# write, step 1 lists them, and terraform destroy stops on the bucket after
# deleting everything else.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ------------------------------------------------------------------------------
# STEP 1: THE MCP CONNECTOR
# ------------------------------------------------------------------------------
if [ -f 02-mcp/terraform.tfstate ]; then
  # apply.sh wrote these inputs; rebuild them from phase 1 if they are gone.
  # Only region and prefix are needed, and destroy deletes what state records.
  if [ ! -f 02-mcp/deployment.tfvars.json ]; then
    echo "NOTE: Reconstructing 02-mcp/deployment.tfvars.json..."
    terraform -chdir=01-infrastructure output -json demo_config \
      | jq '{region: .region, prefix: .prefix}' > 02-mcp/deployment.tfvars.json
  fi
  echo "NOTE: Destroying the MCP connector..."
  terraform -chdir=02-mcp init -input=false > /dev/null
  terraform -chdir=02-mcp destroy -auto-approve -input=false -var-file=deployment.tfvars.json
fi

if [ ! -f 01-infrastructure/terraform.tfstate ]; then
  echo "NOTE: 01-infrastructure has no local state; nothing was deployed from this checkout."
  exit 0
fi

# ------------------------------------------------------------------------------
# CONFIGURATION FOR THE CLEANUP STEP
# ------------------------------------------------------------------------------
# Regenerated from state rather than trusted from disk: the file is
# gitignored and may be missing or stale.
terraform -chdir=01-infrastructure init -input=false > /dev/null
if terraform -chdir=01-infrastructure output -json demo_config > demo-config.json.tmp 2>/dev/null \
    && jq -e .bucket demo-config.json.tmp > /dev/null 2>&1; then
  mv demo-config.json.tmp demo-config.json
else
  rm -f demo-config.json.tmp
  echo "ERROR: Could not read the demo_config output from Terraform state."
  echo "ERROR: Without it the endpoint cannot be located. Fix the state, or delete the"
  echo "ERROR: endpoint by hand (SageMaker console, prefix shown in terraform.tfvars),"
  echo "ERROR: then run: terraform -chdir=01-infrastructure destroy"
  exit 1
fi

# Cleanup needs only boto3. Use the project venv if there is one; otherwise
# a throwaway venv with just boto3, so destroy never depends on the
# scikit-learn Python-version constraints.
PYTHON=".venv/bin/python"
if [ ! -x "${PYTHON}" ] || ! "${PYTHON}" -c "import boto3" 2>/dev/null; then
  PYTHON="python3"
  if ! python3 -c "import boto3" 2>/dev/null; then
    echo "NOTE: Creating .venv-cleanup with boto3 for the cleanup step..."
    python3 -m venv .venv-cleanup
    .venv-cleanup/bin/pip install --quiet "$(grep '^boto3==' requirements.txt)"
    PYTHON=".venv-cleanup/bin/python"
  fi
fi

# ------------------------------------------------------------------------------
# STEP 2: PYTHON-CREATED RESOURCES
# ------------------------------------------------------------------------------
echo "NOTE: Deleting training, model and endpoint resources created by Python..."
if ! "${PYTHON}" demo.py cleanup --purge-data; then
  echo "ERROR: Cleanup failed; Terraform was NOT run, so the role the endpoint"
  echo "ERROR: uses still exists. Fix the error above and run ./destroy.sh again."
  exit 1
fi

# ------------------------------------------------------------------------------
# STEP 3: TERRAFORM-OWNED RESOURCES
# ------------------------------------------------------------------------------
echo "NOTE: Destroying the notebook instance, IAM roles and bucket..."
if ! terraform -chdir=01-infrastructure destroy -auto-approve -input=false; then
  BUCKET="$(jq -r .bucket demo-config.json)"
  echo "ERROR: terraform destroy did not finish. If the error is BucketNotEmpty,"
  echo "ERROR: s3://${BUCKET} holds objects this demo did not write (listed above)."
  echo "ERROR: Move or delete them yourself, then run ./destroy.sh again."
  exit 1
fi

rm -f demo-config.json
echo "NOTE: Infrastructure teardown complete."
