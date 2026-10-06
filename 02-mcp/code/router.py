# ================================================================================
# router.py
#
# Lambda entrypoint for the MCP connector's front door. Dispatches the small set
# of public HTTP routes to the OAuth proxy (oauth.py) and the MCP JSON-RPC
# handler (mcp.py). Every route here is public at the API Gateway layer — auth is
# enforced inside the handlers:
#   - the OAuth endpoints ARE the authentication (they broker Cognito login)
#   - POST /mcp validates the Bearer as a Cognito access token in mcp.py
#
# Routes:
#   GET  /.well-known/oauth-authorization-server  → oauth.oauth_metadata
#   POST /oauth/register                          → oauth.oauth_register
#   GET  /authorize                               → oauth.oauth_authorize
#   GET  /oauth/callback                          → oauth.oauth_callback
#   POST /oauth/token                             → oauth.oauth_token
#   POST /mcp                                      → mcp.handle_mcp
#
# Copied unchanged from aws-cognito-mcp, where this OAuth flow is proven against
# claude.ai. Only mcp.py is specific to this project.
# ================================================================================

import json
import logging

from oauth import (
    oauth_metadata,
    oauth_register,
    oauth_authorize,
    oauth_callback,
    oauth_token,
)
from mcp import handle_mcp

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    method = event["requestContext"]["http"]["method"]
    path   = event["rawPath"]

    logger.info("Router request: %s %s", method, path)

    try:
        if method == "GET" and path == "/.well-known/oauth-authorization-server":
            return oauth_metadata(event)

        if method == "POST" and path == "/oauth/register":
            return oauth_register(event)

        if method == "GET" and path == "/authorize":
            return oauth_authorize(event)

        if method == "GET" and path == "/oauth/callback":
            return oauth_callback(event)

        if method == "POST" and path == "/oauth/token":
            return oauth_token(event)

        if method == "POST" and path == "/mcp":
            return handle_mcp(event)

        return {"statusCode": 404, "body": json.dumps({"error": "not found"})}

    except Exception:
        logger.exception("Unhandled exception")
        return {"statusCode": 500, "body": json.dumps({"error": "internal server error"})}
