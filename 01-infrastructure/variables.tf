# ==============================================================================
# Inputs -- copy terraform.tfvars.example to terraform.tfvars to change them
# ==============================================================================

variable "region" {
  type    = string
  default = "us-east-1"

  # The scikit-learn image lives in a different AWS-owned ECR account per
  # Region (locals.tf). Failing in plan beats a training job that cannot
  # pull its image.
  validation {
    condition     = contains(keys(local.sklearn_registry), var.region)
    error_message = "No SageMaker scikit-learn image registry is known for this Region; see sklearn_registry in locals.tf."
  }
}

variable "prefix" {
  type    = string
  default = "sagemaker-pp"

  # Every resource name starts with this, including the ones Python creates,
  # which is how cleanup finds them. SageMaker names allow 63 characters and
  # the longest suffix added is 22, so 20 leaves headroom.
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,19}$", var.prefix))
    error_message = "prefix: 3-20 characters, lowercase letters, digits and hyphens, starting with a letter."
  }
}

variable "notebook_instance_type" {
  type    = string
  default = "ml.t3.medium"
}

variable "notebook_volume_gb" {
  type    = number
  default = 10
}

variable "training_instance_type" {
  type    = string
  default = "ml.m5.large"
}

# The hard ceiling on one training job's billed runtime. The job itself
# takes about a minute of script time; 15 minutes covers a slow start.
variable "training_max_runtime_seconds" {
  type    = number
  default = 900

  validation {
    condition     = var.training_max_runtime_seconds >= 60 && var.training_max_runtime_seconds <= 3600
    error_message = "training_max_runtime_seconds: between 60 and 3600."
  }
}

variable "serverless_memory_mb" {
  type    = number
  default = 2048

  validation {
    condition     = contains([1024, 2048, 3072, 4096, 5120, 6144], var.serverless_memory_mb)
    error_message = "serverless_memory_mb: 1024, 2048, 3072, 4096, 5120 or 6144."
  }
}

# Concurrent invocations the endpoint will serve before throttling. Also the
# upper bound on how many copies of it can be billed at once.
variable "serverless_max_concurrency" {
  type    = number
  default = 1

  validation {
    condition     = var.serverless_max_concurrency >= 1 && var.serverless_max_concurrency <= 5
    error_message = "serverless_max_concurrency: 1 to 5 for this demo."
  }
}
