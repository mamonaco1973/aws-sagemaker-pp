# ================================================================================
# mcp.py
#
# MCP (Model Context Protocol) server over HTTP, exposing the random forest as a
# tool. Streamable-HTTP transport: a plain POST carrying synchronous JSON-RPC 2.0,
# which is what claude.ai and Claude Desktop speak to a remote connector.
#
# Auth: Bearer token in the Authorization header -- a Cognito access token from
# the OAuth flow in oauth.py, validated via Cognito's /oauth2/userInfo.
#
# One tool, predict_equipment_failure. On tools/call this Lambda calls
# SageMaker Runtime InvokeEndpoint on <prefix>-endpoint with its own IAM role.
# The MCP client never sees the model, the endpoint or any AWS credential --
# only the tool's description, the arguments it sends, and the text returned.
#
# The endpoint is created by Python (demo.py deploy / the notebook), not by
# Terraform, so it may not exist. That is reported to the caller as a tool
# error with the fix, not as a server failure.
# ================================================================================

import json
import logging
import os
import secrets
import time
import urllib.request

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MCP_VERSION  = "2025-03-26"
_SERVER_NAME = "sagemaker-equipment-failure"
_SERVER_VER  = "1.0.0"

_ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "sagemaker-pp-endpoint")

# API Gateway HTTP APIs give an integration 30 seconds. A serverless cold start
# plus retries has to fit inside that, with room to answer.
_CALL_DEADLINE_SECONDS = 24
_RETRY_SECONDS         = 4
_runtime = boto3.client(
    "sagemaker-runtime",
    config=Config(read_timeout=20, connect_timeout=5, retries={"max_attempts": 1}),
)

MAX_READINGS = 100

# GetUser authenticates with the access token itself; no IAM permission needed.
_cognito = boto3.client("cognito-idp")

TOOL = {
    "name": "predict_equipment_failure",
    "description": (
        "Predict whether industrial machines will fail, from three sensor readings "
        "each. Uses a scikit-learn random forest (200 trees) served from an Amazon "
        "SageMaker Serverless Inference endpoint. Returns, per reading, the "
        "predicted label ('healthy' or 'failure') and the probability of failure. "
        "The model was trained on SYNTHETIC demonstration data, so its answers "
        "illustrate the workflow and are not a validated maintenance prediction. "
        "Typical ranges in the training data: temperature 50-90 C, vibration "
        "1-8 mm/s, operating_hours 0-12000."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "readings": {
                "type": "array",
                "description": "One to 100 machines to score.",
                "minItems": 1,
                "maxItems": MAX_READINGS,
                "items": {
                    "type": "object",
                    "properties": {
                        "temperature":     {"type": "number", "description": "Degrees Celsius"},
                        "vibration":       {"type": "number", "description": "RMS vibration, mm/s"},
                        "operating_hours": {"type": "number", "description": "Hours in service"},
                    },
                    "required": ["temperature", "vibration", "operating_hours"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["readings"],
    },
}


# ================================================================================
# HTTP + JSON-RPC helpers
# ================================================================================

def _ok(body, extra_headers=None):
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    return {"statusCode": 200, "headers": headers, "body": json.dumps(body)}


def _accepted():
    # Correct HTTP response for a JSON-RPC notification (no response body).
    return {"statusCode": 202, "headers": {}, "body": ""}


def _http_err(msg, status):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": msg}),
    }


def _rpc_error(req_id, code, message, extra_headers=None):
    return _ok(
        {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}},
        extra_headers,
    )


def _rpc_ok(req_id, result, extra_headers=None):
    return _ok({"jsonrpc": "2.0", "id": req_id, "result": result}, extra_headers)


def _tool_text(req_id, text, is_error=False):
    # Tool failures go back as a result with isError, not a JSON-RPC error, so
    # the model reads the reason and can correct its arguments.
    result = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return _rpc_ok(req_id, result)


# ================================================================================
# Auth -- validate the Cognito access token via userInfo
# ================================================================================

_cognito_userinfo_url = None


def _get_cognito_userinfo_url():
    """Build the Cognito userInfo URL once and cache it in module memory."""
    global _cognito_userinfo_url
    if not _cognito_userinfo_url:
        domain = os.environ.get("COGNITO_DOMAIN", "")
        region = boto3.session.Session().region_name
        _cognito_userinfo_url = (
            f"https://{domain}.auth.{region}.amazoncognito.com/oauth2/userInfo"
        )
    return _cognito_userinfo_url


def _resolve_cognito_token(token):
    """Return the user's email for a valid Cognito access token, else None.

    userInfo is stateless: Cognito checks the signature and expiry, so no
    crypto library is needed here.
    """
    req = urllib.request.Request(
        _get_cognito_userinfo_url(),
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 - fixed Cognito userInfo endpoint, not user-controlled
            claims = json.loads(resp.read())
        return claims.get("email", "").lower().strip() or None
    except Exception:
        pass
    # userInfo accepts only tokens with the openid scope, which the Hosted UI
    # (the claude.ai flow) issues. Tokens from the IAM-gated admin sign-in
    # (validate.sh's throwaway user) carry aws.cognito.signin.user.admin
    # instead; Cognito's GetUser verifies those. Either way Cognito checks the
    # signature and expiry, and the admin flow needs AWS credentials to use.
    try:
        user = _cognito.get_user(AccessToken=token)
        attrs = {a["Name"]: a["Value"] for a in user.get("UserAttributes", [])}
        return attrs.get("email", "").lower().strip() or None
    except Exception:
        return None


def _get_auth_user(event):
    headers = event.get("headers") or {}
    auth    = headers.get("authorization") or headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        return None
    return _resolve_cognito_token(auth[7:].strip())


# ================================================================================
# The tool
# ================================================================================

class ToolError(Exception):
    """A problem the caller (the model) should be told about in plain words."""


def _invoke_endpoint(readings):
    """Send {"instances": readings} to the endpoint; return its JSON response.

    Retries the 502/503/504 a cold serverless container can return while its
    server starts, within the API Gateway time budget.
    """
    deadline = time.time() + _CALL_DEADLINE_SECONDS
    body     = json.dumps({"instances": readings}).encode()
    while True:
        try:
            response = _runtime.invoke_endpoint(
                EndpointName=_ENDPOINT_NAME,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
            return json.loads(response["Body"].read())
        except ClientError as e:
            error  = e.response["Error"]
            status = e.response.get("OriginalStatusCode")
            logger.warning("InvokeEndpoint %s failed: %s %s (original status %s)",
                           _ENDPOINT_NAME, error["Code"], error["Message"], status)
            if error["Code"] == "ModelError" and status == 400:
                # inference.py's validation message, e.g. "reading is missing ...".
                try:
                    detail = json.loads(e.response.get("OriginalMessage", ""))["error"]
                except (ValueError, KeyError, TypeError):
                    detail = e.response.get("OriginalMessage") or error["Message"]
                raise ToolError("The model rejected the input: %s" % detail)
            # Cold start: nginx answers 502/503/504 before the model server is
            # up, or SageMaker reports status 0, "could not get a response".
            if error["Code"] == "ModelError" and status in (0, None, 502, 503, 504):
                if time.time() + _RETRY_SECONDS < deadline:
                    logger.info("Endpoint returned %s (cold start); retrying", status)
                    time.sleep(_RETRY_SECONDS)
                    continue
                raise ToolError(
                    "The model endpoint is still starting up (serverless cold "
                    "start). Try the same call again in a few seconds.")
            if error["Code"].startswith("Validation") and "not found" in error["Message"].lower():
                raise ToolError(
                    "The model endpoint %s is not deployed right now. Deploy it "
                    "with `python demo.py deploy` (or section 9 of the notebook), "
                    "then try again." % _ENDPOINT_NAME)
            if error["Code"] == "ThrottlingException":
                raise ToolError("The endpoint is busy (its concurrency limit is low "
                                "by design). Try again in a moment.")
            raise


def _format(readings, predictions):
    lines = []
    for i, (reading, p) in enumerate(zip(readings, predictions), 1):
        lines.append(
            "%d. temperature=%s C, vibration=%s mm/s, operating_hours=%s -> %s "
            "(failure probability %.1f%%)"
            % (i, reading.get("temperature"), reading.get("vibration"),
               reading.get("operating_hours"), p["label"].upper(),
               100.0 * p["failure_probability"]))
    lines.append("")
    lines.append("Model: random forest on synthetic demonstration data, served by "
                 "SageMaker endpoint %s." % _ENDPOINT_NAME)
    return "\n".join(lines)


def _handle_tools_call(req, user_email):
    req_id    = req.get("id")
    params    = req.get("params") or {}
    tool_name = params.get("name", "")
    if tool_name != TOOL["name"]:
        return _rpc_error(req_id, -32601, f"Unknown tool: {tool_name}")

    readings = (params.get("arguments") or {}).get("readings")
    if not isinstance(readings, list) or not readings:
        return _tool_text(req_id, '"readings" must be a non-empty list of '
                          '{temperature, vibration, operating_hours} objects.', True)
    if len(readings) > MAX_READINGS:
        return _tool_text(req_id, "At most %d readings per call." % MAX_READINGS, True)

    logger.info("tools/call %s: %d reading(s) user=%s", tool_name, len(readings), user_email)
    try:
        # Field validation (missing, unknown, non-numeric, non-finite) is the
        # endpoint's job: inference.py already does it and answers 400.
        response = _invoke_endpoint(readings)
    except ToolError as e:
        return _tool_text(req_id, str(e), is_error=True)
    except Exception:
        logger.exception("InvokeEndpoint failed")
        return _tool_text(req_id, "The prediction service failed unexpectedly; "
                          "see the router Lambda's CloudWatch logs.", is_error=True)
    return _tool_text(req_id, _format(readings, response["predictions"]))


# ================================================================================
# JSON-RPC method handlers
# ================================================================================

def _handle_initialize(req, session_id):
    # Mcp-Session-Id is required by the 2025-03-26 streamable-HTTP transport.
    return _rpc_ok(
        req.get("id"),
        {
            "protocolVersion": MCP_VERSION,
            "capabilities":    {"tools": {}},
            "serverInfo":      {"name": _SERVER_NAME, "version": _SERVER_VER},
        },
        extra_headers={"Mcp-Session-Id": session_id},
    )


def handle_mcp(event):
    """MCP JSON-RPC endpoint -- auth via the OAuth Cognito access token."""
    user_id = _get_auth_user(event)
    if not user_id:
        return _http_err("Unauthorized -- connect via the MCP client's OAuth flow", 401)

    try:
        req = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _http_err("Invalid JSON", 400)

    method = req.get("method", "")
    logger.info("MCP request: method=%s user=%s", method, user_id)

    # Stateless session: a fresh id on initialize, echoed back afterwards.
    session_id = (event.get("headers") or {}).get("mcp-session-id") or secrets.token_hex(16)

    if method == "initialize":
        return _handle_initialize(req, session_id)
    if method in ("notifications/initialized", "notifications/cancelled"):
        return _accepted()
    if method == "tools/list":
        return _rpc_ok(req.get("id"), {"tools": [TOOL]})
    if method == "tools/call":
        return _handle_tools_call(req, user_id)
    if method == "ping":
        return _rpc_ok(req.get("id"), {})

    return _rpc_error(req.get("id"), -32601, f"Method not found: {method}")
