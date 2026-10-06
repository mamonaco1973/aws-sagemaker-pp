#!/bin/bash
# ==============================================================================
# run_local_checks.sh - everything that can be verified without AWS
# ==============================================================================
# Needs Docker and Terraform. Makes no AWS calls and creates nothing in AWS.
#
#   1. Terraform: fmt, validate, and the lifecycle template rendered + bash -n
#   2. Shell scripts: bash -n
#   3. pytest in a Python 3.12 image matching the training container
#   4. train.py through the container's own training entry point (offline)
#   5. inference.py through the container's own serving code (offline,
#      proving the endpoint needs no internet access to start)
#   6. the notebook's offline cells in a Python 3.10 image built from
#      requirements.txt, loading the model the 3.12 run produced
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "NOTE: [1/6] Terraform fmt and validate..."
for phase in 01-infrastructure 02-mcp; do
  terraform -chdir="${phase}" fmt -check -recursive
  terraform -chdir="${phase}" init -backend=false -input=false > /dev/null
  terraform -chdir="${phase}" validate -no-color
done
RENDERED="$(mktemp)"
echo 'templatefile("scripts/on-start.sh", {project_dir = "/home/ec2-user/SageMaker/aws-sagemaker-pp", bucket = "sagemaker-pp-test", region = "us-east-1", config_json = jsonencode({region = "us-east-1"})})' \
  | terraform -chdir=01-infrastructure console | sed '1d;$d' > "${RENDERED}"
bash -n "${RENDERED}"
grep -q 'SOURCE="s3://sagemaker-pp-test/notebook-source"' "${RENDERED}"
rm -f "${RENDERED}"
echo "NOTE: lifecycle script renders and parses."

echo "NOTE: [2/6] Shell syntax..."
for script in *.sh tests/*.sh; do bash -n "${script}"; done

echo "NOTE: Building the test images (cached after the first run)..."
docker build -q -t aws-sagemaker-pp-container:test -f tests/docker/container.Dockerfile . > /dev/null
docker build -q -t aws-sagemaker-pp-kernel:test -f tests/docker/kernel.Dockerfile . > /dev/null

RUN=(docker run --rm --network none -v "${PWD}:/w" -w /w
     -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e PYTHONWARNINGS=ignore)
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

echo "NOTE: [3/6] pytest (Python 3.12, container pins)..."
"${RUN[@]}" aws-sagemaker-pp-container:test python -m pytest -q -p no:cacheprovider tests

echo "NOTE: [4/6] Training through the container's entry point..."
# algo-1 is the hostname SageMaker gives a single training instance.
"${RUN[@]}" --add-host algo-1:127.0.0.1 -v "${OUT}:/out" aws-sagemaker-pp-container:test \
  bash -c "python tests/smoke_training_stack.py | grep -v '^METRIC' && cp /tmp/model.tar.gz /out/"

echo "NOTE: [5/6] Serving through the container's serving code..."
"${RUN[@]}" aws-sagemaker-pp-container:test python tests/smoke_serving_stack.py 2>&1 \
  | grep -E '^NOTE|Error|error' | grep -v '^METRIC'

echo "NOTE: [6/6] Notebook offline cells in the Python 3.10 kernel image..."
"${RUN[@]}" -v "${OUT}:/out:ro" aws-sagemaker-pp-kernel:test \
  python tests/smoke_notebook_offline.py /out/model.tar.gz | grep '^NOTE'

echo "NOTE: All local checks passed."
