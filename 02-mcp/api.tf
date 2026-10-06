# ==============================================================================
# HTTP API -- every route is public; the Lambda enforces auth
# ==============================================================================
# A gateway authorizer would reject the OAuth handshake and the client's first
# unauthenticated /mcp probe before a token exists.

resource "aws_apigatewayv2_api" "mcp" {
  name          = "${local.name}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "router" {
  api_id                 = aws_apigatewayv2_api.mcp.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.router.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

locals {
  routes = [
    "GET /.well-known/oauth-authorization-server",
    "POST /oauth/register",
    "GET /authorize",
    "GET /oauth/callback",
    "POST /oauth/token",
    "POST /mcp",
  ]
}

resource "aws_apigatewayv2_route" "router" {
  for_each  = toset(local.routes)
  api_id    = aws_apigatewayv2_api.mcp.id
  route_key = each.value
  target    = "integrations/${aws_apigatewayv2_integration.router.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.mcp.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "api" {
  statement_id  = "AllowAPIGatewayInvokeRouter"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.router.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.mcp.execution_arn}/*/*"
}
