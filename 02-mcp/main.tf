# ==============================================================================
# Provider Configuration -- phase 2: the MCP connector
# ==============================================================================
# A remote MCP server that lets Claude (claude.ai, Claude Desktop) call the
# model as a tool, signed in through Cognito. Adapted from aws-cognito-mcp:
# oauth.py and router.py are that project's, unchanged; mcp.py is new.
#
# Inputs come from phase 1 (apply.sh writes deployment.tfvars.json). The
# SageMaker endpoint itself is created by Python, so this phase only needs
# its name, which is derived from the prefix.

terraform {
  required_version = ">= 1.9, < 2.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
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

locals {
  name = "${var.prefix}-mcp"
  # Must match Config.endpoint_name in sagemaker_demo/config.py.
  endpoint_name = "${var.prefix}-endpoint"
  endpoint_arn  = "arn:${data.aws_partition.current.partition}:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:endpoint/${local.endpoint_name}"
}

# A random suffix keeps the Cognito Hosted UI domain globally unique.
resource "random_id" "suffix" {
  byte_length = 4
}

data "archive_file" "router" {
  type        = "zip"
  source_dir  = "${path.module}/code"
  output_path = "${path.module}/router.zip"
  excludes    = ["__pycache__"]
}
