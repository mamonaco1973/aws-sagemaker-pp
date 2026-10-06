# ==============================================================================
# Router Lambda -- OAuth endpoints and the MCP tool
# ==============================================================================

resource "aws_iam_role" "router" {
  name = "${local.name}-router-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Principal = { Service = "lambda.amazonaws.com" }
      Effect    = "Allow"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "router_logs" {
  role       = aws_iam_role.router.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# The tool's whole AWS footprint: invoke one endpoint, and the OAuth table.
resource "aws_iam_role_policy" "router" {
  name = "${local.name}-router"
  role = aws_iam_role.router.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sagemaker:InvokeEndpoint"]
        Resource = local.endpoint_arn
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
        Resource = aws_dynamodb_table.oauth_state.arn
      },
    ]
  })
}

resource "aws_cloudwatch_log_group" "router" {
  name              = "/aws/lambda/${local.name}-router"
  retention_in_days = 7
}

resource "aws_lambda_function" "router" {
  function_name    = "${local.name}-router"
  role             = aws_iam_role.router.arn
  runtime          = "python3.14"
  handler          = "router.lambda_handler"
  filename         = data.archive_file.router.output_path
  source_code_hash = data.archive_file.router.output_base64sha256
  # API Gateway HTTP APIs stop waiting at 30s; mcp.py budgets 24s for a cold
  # endpoint, including retries.
  timeout     = 29
  memory_size = 256

  environment {
    variables = {
      TABLE_NAME        = aws_dynamodb_table.oauth_state.name
      COGNITO_DOMAIN    = aws_cognito_user_pool_domain.mcp.domain
      MCP_CLIENT_ID     = aws_cognito_user_pool_client.mcp.id
      MCP_CLIENT_SECRET = aws_cognito_user_pool_client.mcp.client_secret
      ENDPOINT_NAME     = local.endpoint_name
    }
  }

  depends_on = [aws_cloudwatch_log_group.router, aws_iam_role_policy.router]
}
