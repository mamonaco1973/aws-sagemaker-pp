"""The MCP router's tool handling, with Cognito and SageMaker Runtime faked."""

import io
import json
import os
import re
import sys

import pytest
from botocore.exceptions import ClientError

from conftest import ROOT

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("TABLE_NAME", "unused")
sys.path.insert(0, os.path.join(ROOT, "02-mcp", "code"))
import mcp  # noqa: E402

GOOD = {"temperature": 81, "vibration": 5.1, "operating_hours": 11000}


class FakeRuntime:
    def __init__(self, errors=(), predictions=None):
        self.errors = list(errors)
        self.bodies = []
        self.predictions = predictions

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        self.bodies.append(json.loads(Body))
        if self.errors:
            raise self.errors.pop(0)
        instances = self.bodies[-1]["instances"]
        predictions = self.predictions or [
            {"label": "failure", "failure_probability": 0.9566} for _ in instances]
        return {"Body": io.BytesIO(json.dumps({"predictions": predictions}).encode())}


def model_error(status, message="<html>502 Bad Gateway</html>"):
    return ClientError({"Error": {"Code": "ModelError", "Message": "x"},
                        "OriginalStatusCode": status, "OriginalMessage": message},
                       "InvokeEndpoint")


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(mcp, "_runtime", fake)
    monkeypatch.setattr(mcp, "_resolve_cognito_token",
                        lambda token: "user@example.com" if token == "good" else None)
    monkeypatch.setattr(mcp.time, "sleep", lambda s: None)
    return fake


def call(method, params=None, token="good"):
    event = {"headers": {"authorization": "Bearer %s" % token} if token else {},
             "body": json.dumps({"jsonrpc": "2.0", "id": 7, "method": method,
                                 "params": params or {}})}
    response = mcp.handle_mcp(event)
    return response["statusCode"], json.loads(response["body"]) if response["body"] else None


def tool_call(readings):
    return call("tools/call", {"name": "predict_equipment_failure",
                               "arguments": {"readings": readings}})


def test_no_or_bad_token_is_401(runtime):
    assert call("tools/list", token=None)[0] == 401
    assert call("tools/list", token="forged")[0] == 401


def test_tools_list_schema_matches_what_the_endpoint_accepts(runtime):
    status, body = call("tools/list")
    (tool,) = body["result"]["tools"]
    item = tool["inputSchema"]["properties"]["readings"]["items"]
    import dataset
    assert sorted(item["required"]) == sorted(dataset.FEATURES)
    assert sorted(item["properties"]) == sorted(dataset.FEATURES)
    assert tool["inputSchema"]["properties"]["readings"]["maxItems"] == 100   # inference.MAX_INSTANCES


def test_tools_call_sends_readings_as_instances_and_formats_the_answer(runtime):
    status, body = tool_call([GOOD, GOOD])
    assert status == 200
    assert runtime.bodies == [{"instances": [GOOD, GOOD]}]
    text = body["result"]["content"][0]["text"]
    assert "isError" not in body["result"]
    assert text.count("FAILURE (failure probability 95.7%)") == 2
    assert "synthetic" in text


def test_validation_400_becomes_a_tool_error_the_model_can_read(runtime):
    runtime.errors = [model_error(400, json.dumps({"error": "reading is missing operating_hours"}))]
    _, body = tool_call([{"temperature": 70, "vibration": 3}])
    assert body["result"]["isError"] is True
    assert "reading is missing operating_hours" in body["result"]["content"][0]["text"]


def test_cold_start_502_is_retried(runtime):
    runtime.errors = [model_error(502), model_error(503), model_error(0)]
    _, body = tool_call([GOOD])
    assert "isError" not in body["result"] and len(runtime.bodies) == 4


def test_undeployed_endpoint_says_how_to_deploy(runtime):
    runtime.errors = [ClientError({"Error": {"Code": "ValidationError",
                                             "Message": "Endpoint sagemaker-pp-endpoint of "
                                                        "account 1 not found."}},
                                  "InvokeEndpoint")]
    _, body = tool_call([GOOD])
    assert body["result"]["isError"] is True
    assert "python demo.py deploy" in body["result"]["content"][0]["text"]


def test_bad_arguments_never_reach_the_endpoint(runtime):
    for readings in (None, [], "hot", [GOOD] * 101):
        _, body = tool_call(readings)
        assert body["result"]["isError"] is True
    assert runtime.bodies == []


def test_unknown_tool_and_method(runtime):
    _, body = call("tools/call", {"name": "rm_rf", "arguments": {}})
    assert body["error"]["code"] == -32601
    _, body = call("resources/list")
    assert body["error"]["code"] == -32601


def test_router_only_reaches_the_demo_endpoint():
    """IAM grants InvokeEndpoint on exactly the endpoint name the demo creates."""
    with open(os.path.join(ROOT, "02-mcp", "main.tf")) as f:
        main = f.read()
    with open(os.path.join(ROOT, "02-mcp", "lambda.tf")) as f:
        lam = f.read()
    assert re.search(r'endpoint_name\s*=\s*"\$\{var\.prefix\}-endpoint"', main)
    assert 'Resource = local.endpoint_arn' in lam
    from sagemaker_demo.config import Config
    assert Config(prefix="sagemaker-pp").endpoint_name == "sagemaker-pp-endpoint"


def test_token_check_falls_back_to_get_user(monkeypatch):
    """validate.sh's admin-flow token lacks openid, so userInfo rejects it."""
    def userinfo_rejects(*a, **k):
        raise OSError("401 Access token does not contain the 'openid' scope")

    class FakeCognito:
        def get_user(self, AccessToken):
            if AccessToken != "admin-token":
                raise ClientError({"Error": {"Code": "NotAuthorizedException",
                                             "Message": "Invalid Access Token"}}, "GetUser")
            return {"UserAttributes": [{"Name": "email", "Value": "Ops@Example.com"}]}

    monkeypatch.setattr(mcp.urllib.request, "urlopen", userinfo_rejects)
    monkeypatch.setattr(mcp, "_cognito", FakeCognito())
    assert mcp._resolve_cognito_token("admin-token") == "ops@example.com"
    assert mcp._resolve_cognito_token("forged") is None
