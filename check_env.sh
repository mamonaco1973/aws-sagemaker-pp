#!/bin/bash
# ==============================================================================
# check_env.sh - Environment Validation
# ------------------------------------------------------------------------------
# Purpose:
#   - Confirms the CLI tools apply.sh and destroy.sh need are on PATH.
#   - Confirms Terraform is new enough and the AWS CLI is authenticated.
# ==============================================================================
set -euo pipefail

echo "NOTE: Validating that required commands are found in your PATH."
commands=("aws" "terraform" "jq" "python3" "curl" "openssl")
all_found=true

for cmd in "${commands[@]}"; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "ERROR: $cmd is not found in the current PATH."
    all_found=false
  else
    echo "NOTE: $cmd is found in the current PATH."
  fi
done

if [ "$all_found" = true ]; then
  echo "NOTE: All required commands are available."
else
  echo "ERROR: One or more commands are missing."
  exit 1
fi

# Variable validation that reads locals (the Region check) needs 1.9.
TF_VERSION="$(terraform version -json | jq -r .terraform_version)"
if [ "$(printf '%s\n1.9.0\n' "${TF_VERSION}" | sort -V | head -1)" != "1.9.0" ]; then
  echo "ERROR: Terraform ${TF_VERSION} is too old; 1.9 or newer is required."
  exit 1
fi
echo "NOTE: Terraform ${TF_VERSION}."

echo "NOTE: Checking AWS cli connection."
if ! aws sts get-caller-identity --query "Account" --output text > /dev/null; then
  echo "ERROR: Failed to connect to AWS. Please check your credentials and environment variables."
  exit 1
fi
echo "NOTE: Successfully logged into AWS."
