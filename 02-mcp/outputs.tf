output "mcp_endpoint" {
  description = "Connector URL: paste into claude.ai as a custom connector"
  value       = "${aws_apigatewayv2_api.mcp.api_endpoint}/mcp"
}

output "api_base_url" { value = aws_apigatewayv2_api.mcp.api_endpoint }
output "cognito_user_pool_id" { value = aws_cognito_user_pool.mcp.id }
output "cognito_domain" { value = aws_cognito_user_pool_domain.mcp.domain }
output "mcp_client_id" { value = aws_cognito_user_pool_client.mcp.id }
output "endpoint_name" { value = local.endpoint_name }
output "router_function" { value = aws_lambda_function.router.function_name }

# Used by validate.sh to compute Cognito's SECRET_HASH for its throwaway user.
output "mcp_client_secret" {
  value     = aws_cognito_user_pool_client.mcp.client_secret
  sensitive = true
}
