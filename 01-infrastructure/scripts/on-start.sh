#!/bin/bash
# ==============================================================================
# Notebook lifecycle script -- runs as root on every notebook start
# ==============================================================================
# Rendered by Terraform (templatefile). Everything happens as ec2-user, under
# /home/ec2-user/SageMaker: that is the only directory a notebook instance
# keeps across stop/start.
#
#   1. copy the project's code from S3 (code always; the notebook only if
#      absent, so your edits and outputs survive a restart)
#   2. write demo-config.json from the Terraform outputs -- the notebook reads
#      this instead of anyone pasting ARNs
#   3. build the pinned "sagemaker-demo" kernel once, in the background (a
#      lifecycle script that runs past 5 minutes fails the notebook start),
#      and re-register it on later starts
#
# Log: CloudWatch /aws/sagemaker/NotebookInstances, stream
# <notebook>/LifecycleConfigOnStart; kernel build: <project>/kernel-setup.log
# ==============================================================================
set -euo pipefail

sudo -u ec2-user -i bash <<'AS_EC2_USER'
set -euo pipefail

PROJECT_DIR="${project_dir}"
SOURCE="s3://${bucket}/notebook-source"
REGION="${region}"
VENV="/home/ec2-user/SageMaker/.venvs/sagemaker-demo"
KERNEL_NAME="sagemaker-demo"

mkdir -p "$PROJECT_DIR/notebook"

# --- 1. Project code ----------------------------------------------------------
# No --delete: data/, artifacts/ and demo-config.json live here too.
aws s3 sync "$SOURCE/" "$PROJECT_DIR/" --region "$REGION" --exclude "notebook/*" --only-show-errors

for key in $(aws s3 ls "$SOURCE/notebook/" --region "$REGION" | awk '{print $4}'); do
  if [ ! -f "$PROJECT_DIR/notebook/$key" ]; then
    aws s3 cp "$SOURCE/notebook/$key" "$PROJECT_DIR/notebook/$key" --region "$REGION" --only-show-errors
  fi
done

# --- 2. Configuration ---------------------------------------------------------
cat > "$PROJECT_DIR/demo-config.json" <<'JSON'
${config_json}
JSON

# --- 3. Kernel ----------------------------------------------------------------
# conda_python3 is Python 3.10 on notebook-al2023-v1, which the pinned
# scikit-learn 1.4.2 supports. A separate virtualenv keeps the pins from
# fighting the packages preinstalled in conda_python3.
BASE_PYTHON="/home/ec2-user/anaconda3/envs/python3/bin/python"
WANT="$(sha256sum "$PROJECT_DIR/requirements.txt" | cut -d' ' -f1)"
HAVE="$(cat "$VENV/.requirements.sha256" 2>/dev/null || true)"

register_kernel() {
  "$VENV/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" \
    --display-name "Python 3 (sagemaker-demo)"
}

if [ "$WANT" = "$HAVE" ]; then
  register_kernel
  echo "kernel $KERNEL_NAME ready (unchanged requirements)"
else
  nohup bash -c "
    set -e
    echo \"building $VENV at \$(date -u)\"
    rm -rf '$VENV'
    '$BASE_PYTHON' -m venv '$VENV'
    '$VENV/bin/pip' install --quiet --upgrade pip
    '$VENV/bin/pip' install --quiet -r '$PROJECT_DIR/requirements.txt'
    '$VENV/bin/python' -m ipykernel install --user --name '$KERNEL_NAME' --display-name 'Python 3 (sagemaker-demo)'
    echo '$WANT' > '$VENV/.requirements.sha256'
    echo \"KERNEL READY at \$(date -u)\"
  " < /dev/null > "$PROJECT_DIR/kernel-setup.log" 2>&1 &
  echo "building kernel $KERNEL_NAME in the background; see $PROJECT_DIR/kernel-setup.log"
fi
AS_EC2_USER
