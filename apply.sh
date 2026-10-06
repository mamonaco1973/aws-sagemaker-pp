#!/bin/bash
# ==============================================================================
# File: apply.sh
# ==============================================================================
# Purpose:
#   Deploys the static infrastructure for the SageMaker random forest demo:
#     1. 01-infrastructure: the S3 bucket, the notebook and execution IAM
#        roles, and the notebook instance with its lifecycle script
#     2. 02-mcp: the MCP connector that lets Claude call the model as a tool
#
# Notes:
#   - Training jobs and endpoints are NOT created here. The notebook (or
#     demo.py) creates them; destroy.sh removes them before Terraform runs.
#   - Region and prefix come from 01-infrastructure/terraform.tfvars if
#     present (copy terraform.tfvars.example), else the defaults.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ------------------------------------------------------------------------------
# ENVIRONMENT PRE-CHECK
# ------------------------------------------------------------------------------
echo "NOTE: Running environment validation..."
./check_env.sh

# ------------------------------------------------------------------------------
# BUILD THE INFRASTRUCTURE
# ------------------------------------------------------------------------------
# The notebook instance takes about five minutes to reach InService, and
# Terraform waits for it.
echo "NOTE: Deploying the bucket, IAM roles and notebook instance..."

terraform -chdir=01-infrastructure init -input=false
terraform -chdir=01-infrastructure validate -no-color
terraform -chdir=01-infrastructure apply -auto-approve -input=false

# ------------------------------------------------------------------------------
# WRITE THE LOCAL CONFIGURATION
# ------------------------------------------------------------------------------
# demo.py reads this on your machine. The notebook instance gets the same
# JSON from its lifecycle script. Gitignored: it names your account.
terraform -chdir=01-infrastructure output -json demo_config > demo-config.json
echo "NOTE: Wrote demo-config.json for demo.py."

# ------------------------------------------------------------------------------
# PHASE 2: THE MCP CONNECTOR
# ------------------------------------------------------------------------------
# Cognito, an HTTP API and one Lambda that exposes the model to Claude as a
# tool. Its inputs come from phase 1 and are kept in a file destroy.sh reuses.
echo "NOTE: Deploying the MCP connector (Cognito, API Gateway, router Lambda)..."

jq '{region: .region, prefix: .prefix}' demo-config.json > 02-mcp/deployment.tfvars.json
terraform -chdir=02-mcp init -input=false
terraform -chdir=02-mcp validate -no-color
terraform -chdir=02-mcp apply -auto-approve -input=false -var-file=deployment.tfvars.json

# ------------------------------------------------------------------------------
# BUILD VALIDATION
# ------------------------------------------------------------------------------
echo "NOTE: Running build validation..."
./validate.sh

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
