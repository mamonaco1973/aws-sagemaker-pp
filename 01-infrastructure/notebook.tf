# ==============================================================================
# Notebook Instance -- a standalone Jupyter server, not SageMaker Studio
# ==============================================================================
# Billed per hour while InService (ml.t3.medium: $0.05/hour in us-east-1).
# Stop it when you are not using it: `python demo.py notebook-stop` or the
# console's Stop button. Files under /home/ec2-user/SageMaker survive a stop.

locals {
  # Everything the notebook needs to find the infrastructure. The lifecycle
  # script writes this as demo-config.json, and apply.sh writes the same
  # JSON on your machine for demo.py.
  demo_config = {
    project                      = "aws-sagemaker-pp"
    region                       = var.region
    prefix                       = var.prefix
    bucket                       = aws_s3_bucket.demo.id
    execution_role_arn           = aws_iam_role.execution.arn
    notebook_name                = "${local.name}-notebook"
    sklearn_image_uri            = local.sklearn_image_uri
    sklearn_framework_version    = local.sklearn_framework_version
    training_instance_type       = var.training_instance_type
    training_max_runtime_seconds = var.training_max_runtime_seconds
    serverless_memory_mb         = var.serverless_memory_mb
    serverless_max_concurrency   = var.serverless_max_concurrency
  }
}

resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "demo" {
  name = "${local.name}-on-start"
  on_start = base64encode(templatefile("${path.module}/scripts/on-start.sh", {
    project_dir = local.notebook_project_dir
    bucket      = aws_s3_bucket.demo.id
    region      = var.region
    config_json = jsonencode(local.demo_config)
  }))
}

resource "aws_sagemaker_notebook_instance" "demo" {
  name                  = "${local.name}-notebook"
  instance_type         = var.notebook_instance_type
  role_arn              = aws_iam_role.notebook.arn
  platform_identifier   = "notebook-al2023-v1"
  volume_size           = var.notebook_volume_gb
  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.demo.name

  # Public internet straight from the instance (pip installs the kernel),
  # without a VPC, so no NAT gateway. Jupyter itself is reached only through
  # the console's presigned URL, never a public port.
  direct_internet_access = "Enabled"
  root_access            = "Disabled"

  instance_metadata_service_configuration {
    minimum_instance_metadata_service_version = "2"
  }

  # The role's policy must exist before the lifecycle script runs aws s3 sync.
  # The project files must be in S3 before it copies them.
  depends_on = [aws_iam_role_policy.notebook, aws_s3_object.notebook_source]
}
