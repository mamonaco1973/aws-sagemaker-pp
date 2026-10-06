# ==============================================================================
# Cognito -- who may call the model through the connector
# ==============================================================================
# Claude signs the user in through the Hosted UI (brokered by oauth.py) and
# sends the resulting access token on every POST /mcp; mcp.py validates it.

resource "aws_cognito_user_pool" "mcp" {
  name = "${local.name}-users-${random_id.suffix.hex}"

  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]

  password_policy {
    minimum_length    = 12
    require_lowercase = true
    require_uppercase = true
    require_numbers   = true
    require_symbols   = false
  }

  schema {
    name                = "email"
    attribute_data_type = "String"
    required            = true
    mutable             = true
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  # Open self sign-up: anyone who reaches the Hosted UI can register (with
  # email verification) and call the model. Acceptable for this demo -- the
  # model is synthetic and a prediction costs a fraction of a cent. Set to true
  # to allow only users created with ./add_mcp_user.sh.
  admin_create_user_config {
    allow_admin_create_user_only = false
  }
}

resource "aws_cognito_user_pool_domain" "mcp" {
  domain       = "${local.name}-${random_id.suffix.hex}"
  user_pool_id = aws_cognito_user_pool.mcp.id
}

# Confidential client: the secret lives only in the router Lambda's
# environment. Only our own /oauth/callback is registered -- claude.ai's
# per-organisation redirect_uri is brokered by oauth.py.
resource "aws_cognito_user_pool_client" "mcp" {
  name         = "${local.name}-${random_id.suffix.hex}"
  user_pool_id = aws_cognito_user_pool.mcp.id

  generate_secret = true

  # ADMIN_USER_PASSWORD_AUTH lets validate.sh get a token for a throwaway user
  # without a browser. It is an IAM-authorized admin API, not reachable by
  # anyone holding only the connector URL.
  explicit_auth_flows = ["ALLOW_REFRESH_TOKEN_AUTH", "ALLOW_ADMIN_USER_PASSWORD_AUTH"]

  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "profile"]

  # Claude keeps the access token for the session and does not refresh it, so
  # issue the 24h maximum; oauth_token() reports the same lifetime.
  access_token_validity = 24
  token_validity_units {
    access_token = "hours"
  }

  supported_identity_providers = ["COGNITO"]
  callback_urls                = ["${aws_apigatewayv2_api.mcp.api_endpoint}/oauth/callback"]
}
