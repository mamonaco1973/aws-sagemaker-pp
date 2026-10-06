# Transient OAuth state for oauth.py: PENDINGAUTH (the client's redirect_uri
# while the user signs in) and AUTHCODE (one-time code -> Cognito token).
# Both expire after 5 minutes through DynamoDB TTL.
resource "aws_dynamodb_table" "oauth_state" {
  name         = "${local.name}-oauth-${random_id.suffix.hex}"
  billing_mode = "PAY_PER_REQUEST"

  hash_key  = "pk"
  range_key = "sk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}
