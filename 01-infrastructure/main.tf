# ==============================================================================
# Provider Configuration
# ==============================================================================
# Terraform owns the static pieces only: the bucket, two IAM roles, and the
# notebook instance. Training jobs, models, endpoint configs and endpoints
# are created by Python (sagemaker_demo/) and removed by `demo.py cleanup`,
# which destroy.sh runs before `terraform destroy`.

terraform {
  required_version = ">= 1.9, < 2.0" # validation that reads locals needs 1.9
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "aws-sagemaker-pp", Prefix = var.prefix, ManagedBy = "Terraform" }
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
