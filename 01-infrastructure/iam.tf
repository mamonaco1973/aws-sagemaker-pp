# ==============================================================================
# IAM -- two roles, two jobs
# ==============================================================================
#   notebook role    what the notebook's code may ask SageMaker to do
#   execution role   what the training job and the endpoint run as
#
# They are separate on purpose. The notebook starts jobs and deploys
# endpoints but never touches the container image; the containers read data
# and write artifacts but cannot start or delete anything. The notebook
# hands the execution role to SageMaker through iam:PassRole, and that is
# the only role it may hand over.

locals {
  account = data.aws_caller_identity.current.account_id
  arn_sm  = "arn:${data.aws_partition.current.partition}:sagemaker:${var.region}:${local.account}"
  arn_log = "arn:${data.aws_partition.current.partition}:logs:${var.region}:${local.account}:log-group"

  # Name patterns the Python side uses (sagemaker_demo/workflow.py).
  training_jobs_arn    = "${local.arn_sm}:training-job/${local.name}-train-*"
  models_arn           = "${local.arn_sm}:model/${local.name}-model-*"
  endpoint_configs_arn = "${local.arn_sm}:endpoint-config/${local.name}-epc-*"
  endpoint_arn         = "${local.arn_sm}:endpoint/${local.name}-endpoint"
  notebook_arn         = "${local.arn_sm}:notebook-instance/${local.name}-notebook"
}

data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

# ------------------------------------------------------------------------------
# Execution role: training jobs and the serverless endpoint
# ------------------------------------------------------------------------------
resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "execution" {
  # Training reads its channels and its source; the endpoint reads the
  # model artifact. Only training writes, and only under training-output/.
  statement {
    sid     = "ReadInputs"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.demo.arn}/data/*",
      "${aws_s3_bucket.demo.arn}/code/*",
      "${aws_s3_bucket.demo.arn}/training-output/*",
    ]
  }
  statement {
    sid       = "WriteArtifacts"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.demo.arn}/training-output/*"]
  }
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.demo.arn]
  }

  # Pull the AWS-owned scikit-learn image. GetAuthorizationToken has no
  # resource-level scoping; the pulls are limited to that one repository.
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "PullSklearnImage"
    actions   = ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    resources = ["arn:${data.aws_partition.current.partition}:ecr:${var.region}:${local.sklearn_registry[var.region]}:repository/${local.sklearn_repository}"]
  }

  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
    ]
    resources = ["${local.arn_log}:/aws/sagemaker/*"]
  }
  # Instance and endpoint metrics. PutMetricData has no resource ARNs; the
  # namespace condition keeps it to SageMaker's own namespaces.
  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*", "aws/sagemaker/*"]
    }
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "${local.name}-execution"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}

# ------------------------------------------------------------------------------
# Notebook role: the notebook's code drives SageMaker with this
# ------------------------------------------------------------------------------
resource "aws_iam_role" "notebook" {
  name               = "${local.name}-notebook-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "notebook" {
  # Create, watch and delete the demo's own resources -- by name prefix, so
  # the notebook cannot touch any other SageMaker resource in the account.
  statement {
    sid = "TrainingJobs"
    actions = [
      "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob", "sagemaker:AddTags",
    ]
    resources = [local.training_jobs_arn]
  }
  statement {
    sid = "ModelsAndEndpoints"
    actions = [
      "sagemaker:CreateModel", "sagemaker:DescribeModel", "sagemaker:DeleteModel",
      "sagemaker:CreateEndpointConfig", "sagemaker:DescribeEndpointConfig",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:CreateEndpoint", "sagemaker:DescribeEndpoint", "sagemaker:UpdateEndpoint",
      "sagemaker:DeleteEndpoint", "sagemaker:AddTags",
    ]
    resources = [local.models_arn, local.endpoint_configs_arn, local.endpoint_arn]
  }
  # The endpoint has no URL of its own; this permission is what lets a caller
  # reach it at all.
  statement {
    sid       = "Invoke"
    actions   = ["sagemaker:InvokeEndpoint"]
    resources = [local.endpoint_arn]
  }
  statement {
    sid       = "OwnNotebook"
    actions   = ["sagemaker:DescribeNotebookInstance", "sagemaker:StopNotebookInstance"]
    resources = [local.notebook_arn]
  }
  # List calls cannot be scoped to resources; the code filters by prefix.
  statement {
    sid = "List"
    actions = [
      "sagemaker:ListTrainingJobs", "sagemaker:ListModels",
      "sagemaker:ListEndpointConfigs", "sagemaker:ListEndpoints",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "PassExecutionRole"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  statement {
    sid       = "BucketObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.demo.arn}/*"]
  }
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.demo.arn]
  }

  # Read training and endpoint logs from the notebook, write its own
  # lifecycle-script log, and delete this demo's logs during cleanup.
  statement {
    sid = "ReadLogs"
    actions = [
      "logs:DescribeLogStreams", "logs:GetLogEvents", "logs:FilterLogEvents",
    ]
    resources = ["${local.arn_log}:/aws/sagemaker/*"]
  }
  statement {
    sid       = "NotebookLogs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${local.arn_log}:/aws/sagemaker/NotebookInstances*"]
  }
  statement {
    sid     = "CleanupLogs"
    actions = ["logs:DeleteLogGroup", "logs:DeleteLogStream"]
    resources = [
      "${local.arn_log}:/aws/sagemaker/Endpoints/${local.name}-endpoint",
      "${local.arn_log}:/aws/sagemaker/Endpoints/${local.name}-endpoint:*",
      "${local.arn_log}:/aws/sagemaker/TrainingJobs:log-stream:${local.name}-train-*",
    ]
  }
}

resource "aws_iam_role_policy" "notebook" {
  name   = "${local.name}-notebook"
  role   = aws_iam_role.notebook.id
  policy = data.aws_iam_policy_document.notebook.json
}
