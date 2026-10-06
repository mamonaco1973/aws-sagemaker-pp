"""Orchestration against in-memory fakes of SageMaker, S3 and CloudWatch Logs.

No AWS calls. The fakes enforce the one rule that makes ordering matter --
an endpoint config cannot be deleted while an endpoint uses it -- so the
cleanup-order test fails if the order is wrong.
"""

import fnmatch
import json
import os
import re

import pytest
from botocore.exceptions import ClientError

import train
from conftest import ROOT
from sagemaker_demo import cleanup, workflow
from sagemaker_demo.config import Config

PREFIX = "sagemaker-pp"


def not_found(what):
    return ClientError({"Error": {"Code": "ValidationException",
                                  "Message": 'Could not find %s "x".' % what}}, "Describe")


class FakeSageMaker:
    def __init__(self):
        self.models, self.configs, self.endpoints, self.jobs = {}, {}, {}, {}
        self.calls = []

    # -- training --
    def create_training_job(self, **req):
        self.calls.append(("create_training_job", req["TrainingJobName"]))
        self.jobs[req["TrainingJobName"]] = {"TrainingJobStatus": "InProgress", "req": req}

    def describe_training_job(self, TrainingJobName):
        if TrainingJobName not in self.jobs:
            # DescribeTrainingJob's real wording differs from the others.
            raise ClientError({"Error": {"Code": "ValidationException",
                                         "Message": "Requested resource not found."}},
                              "DescribeTrainingJob")
        job = self.jobs[TrainingJobName]
        return {"TrainingJobName": TrainingJobName,
                "TrainingJobStatus": job["TrainingJobStatus"],
                "SecondaryStatus": job["TrainingJobStatus"],
                "ResourceConfig": {"InstanceType": "ml.m5.large"},
                "ModelArtifacts": {"S3ModelArtifacts":
                                   "s3://bucket/training-output/%s/output/model.tar.gz"
                                   % TrainingJobName},
                "FinalMetricDataList": []}

    def list_training_jobs(self, NameContains, StatusEquals=None, **_):
        return {"TrainingJobSummaries": [
            {"TrainingJobName": n, "TrainingJobStatus": j["TrainingJobStatus"]}
            for n, j in sorted(self.jobs.items(), reverse=True)
            if NameContains in n and (StatusEquals is None or j["TrainingJobStatus"] == StatusEquals)]}

    def stop_training_job(self, TrainingJobName):
        self.calls.append(("stop_training_job", TrainingJobName))
        self.jobs[TrainingJobName]["TrainingJobStatus"] = "Stopped"

    # -- models / configs / endpoints --
    def create_model(self, **req):
        self.calls.append(("create_model", req["ModelName"]))
        self.models[req["ModelName"]] = req

    def describe_model(self, ModelName):
        if ModelName not in self.models:
            raise not_found("model")
        return self.models[ModelName]

    def delete_model(self, ModelName):
        self.calls.append(("delete_model", ModelName))
        del self.models[ModelName]

    def list_models(self, NameContains, **_):
        return {"Models": [{"ModelName": n} for n in self.models if NameContains in n]}

    def create_endpoint_config(self, **req):
        self.calls.append(("create_endpoint_config", req["EndpointConfigName"]))
        self.configs[req["EndpointConfigName"]] = req

    def describe_endpoint_config(self, EndpointConfigName):
        if EndpointConfigName not in self.configs:
            raise not_found("endpoint configuration")
        return self.configs[EndpointConfigName]

    def delete_endpoint_config(self, EndpointConfigName):
        if any(e["EndpointConfigName"] == EndpointConfigName for e in self.endpoints.values()):
            raise ClientError({"Error": {"Code": "ValidationException",
                                         "Message": "config in use"}}, "DeleteEndpointConfig")
        self.calls.append(("delete_endpoint_config", EndpointConfigName))
        del self.configs[EndpointConfigName]

    def list_endpoint_configs(self, NameContains, **_):
        return {"EndpointConfigs": [{"EndpointConfigName": n}
                                    for n in self.configs if NameContains in n]}

    def create_endpoint(self, EndpointName, EndpointConfigName, Tags):
        self.calls.append(("create_endpoint", EndpointName))
        self.endpoints[EndpointName] = {"EndpointName": EndpointName, "EndpointStatus": "InService",
                                        "EndpointConfigName": EndpointConfigName}

    def update_endpoint(self, EndpointName, EndpointConfigName):
        self.calls.append(("update_endpoint", EndpointConfigName))
        self.endpoints[EndpointName]["EndpointConfigName"] = EndpointConfigName

    def describe_endpoint(self, EndpointName):
        if EndpointName not in self.endpoints:
            raise not_found("endpoint")
        return dict(self.endpoints[EndpointName])

    def delete_endpoint(self, EndpointName):
        self.calls.append(("delete_endpoint", EndpointName))
        del self.endpoints[EndpointName]

    def list_endpoints(self, NameContains, **_):
        return {"Endpoints": [{"EndpointName": n} for n in self.endpoints if NameContains in n]}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = 0

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject")
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **_):
        self.puts += 1
        self.objects[Key] = Body

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": ""}}, "HeadObject")

    def list_objects_v2(self, Bucket, **_):
        return {"Contents": [{"Key": k} for k in sorted(self.objects)]}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            del self.objects[o["Key"]]


class FakeLogs:
    def delete_log_group(self, logGroupName):
        raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": ""}}, "x")

    def describe_log_streams(self, **_):
        return {"logStreams": []}


@pytest.fixture
def env(monkeypatch):
    cfg = Config(region="test-region-1", prefix=PREFIX, bucket="bucket",
                 execution_role_arn="arn:aws:iam::123456789012:role/sagemaker-pp-execution-role",
                 sklearn_image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/"
                                   "sagemaker-scikit-learn:1.4-2-py312-cpu-py3",
                 training_instance_type="ml.m5.large", training_max_runtime_seconds=900,
                 serverless_memory_mb=2048, serverless_max_concurrency=1)
    sm, s3 = FakeSageMaker(), FakeS3()
    monkeypatch.setitem(workflow._CLIENTS, cfg.region,
                        {"sagemaker": sm, "s3": s3, "logs": FakeLogs(), "runtime": None})
    monkeypatch.setattr(workflow.time, "sleep", lambda s: None)
    return cfg, sm, s3


def completed_job(sm, cfg, stamp):
    name = "%s-train-%s" % (cfg.prefix, stamp)
    sm.jobs[name] = {"TrainingJobStatus": "Completed"}
    return name


# ------------------------------------------------------------------------------
# Contracts between the request and the code that runs in the container
# ------------------------------------------------------------------------------
def test_hyperparameters_are_flags_train_py_accepts(env):
    cfg, _, _ = env
    req = workflow.training_job_request(cfg, "job", "s3://b/code/x.tar.gz",
                                        {"train": "s3://b/data/train/", "test": "s3://b/data/test/"})
    hp = req["HyperParameters"]
    assert json.loads(hp["sagemaker_program"]) == "train.py"
    assert os.path.isfile(os.path.join(ROOT, "ml", json.loads(hp["sagemaker_program"])))
    flags = []
    for key, value in hp.items():
        if not key.startswith("sagemaker_"):
            flags += ["--%s" % key, value]
    args = train.parse_args(flags)
    assert args.n_estimators == 200
    assert all(isinstance(v, str) for v in hp.values()), "SageMaker requires string values"


def test_training_request_uses_config_limits_and_separate_channels(env):
    cfg, _, _ = env
    uris = {"train": "s3://bucket/data/train/", "test": "s3://bucket/data/test/"}
    req = workflow.training_job_request(cfg, "job", "s3://bucket/code/x.tar.gz", uris)
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 900
    assert req["ResourceConfig"]["InstanceType"] == "ml.m5.large"
    assert req["RoleArn"] == cfg.execution_role_arn
    channels = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"]
                for c in req["InputDataConfig"]}
    assert channels == uris


def test_serving_env_points_at_the_code_train_py_ships(env, trained):
    cfg, _, _ = env
    model_dir, _ = trained
    container = workflow.model_request(cfg, "%s-train-1" % PREFIX, "s3://x")["PrimaryContainer"]
    environment = container["Environment"]
    assert container["Image"] == cfg.sklearn_image_uri    # same image as training
    # model.tar.gz is extracted to /opt/ml/model; the code must be where the
    # environment says it is.
    relative = os.path.relpath(environment["SAGEMAKER_SUBMIT_DIRECTORY"], "/opt/ml/model")
    assert os.path.isfile(os.path.join(model_dir, relative, environment["SAGEMAKER_PROGRAM"]))


def test_serverless_config_has_no_provisioned_concurrency(env):
    cfg, _, _ = env
    (variant,) = workflow.endpoint_config_request(cfg, "%s-train-1" % PREFIX)["ProductionVariants"]
    assert variant["ServerlessConfig"] == {"MemorySizeInMB": 2048, "MaxConcurrency": 1}
    assert "ProvisionedConcurrency" not in variant["ServerlessConfig"]
    assert "InstanceType" not in variant


def test_generated_names_fit_iam_patterns_and_length_limits(env):
    """iam.tf grants by name pattern; a drift here would be an AccessDenied in AWS."""
    with open(os.path.join(ROOT, "01-infrastructure", "iam.tf")) as f:
        iam = f.read()
    prefix = "a" * 20                                       # longest allowed prefix
    patterns = {key: re.search(r'%s\s*=\s*"\$\{local\.arn_sm\}:([^"]+)"' % key, iam).group(1)
                .replace("${local.name}", prefix)
                for key in ("training_jobs_arn", "models_arn", "endpoint_configs_arn", "endpoint_arn")}
    cfg = Config(env[0], prefix=prefix)
    job = "%s-train-20261006-142530" % prefix
    names = {"training_jobs_arn": "training-job/" + job,
             "models_arn": "model/" + workflow.model_name_for(cfg, job),
             "endpoint_configs_arn": "endpoint-config/" + workflow.endpoint_config_name_for(cfg, job),
             "endpoint_arn": "endpoint/" + cfg.endpoint_name}
    for key, name in names.items():
        assert fnmatch.fnmatchcase(name, patterns[key]), (name, patterns[key])
        assert len(name.split("/", 1)[1]) <= 63


# ------------------------------------------------------------------------------
# Repeat runs do not duplicate anything
# ------------------------------------------------------------------------------
def test_source_package_is_reproducible(env, monkeypatch):
    cfg, _, s3 = env
    first_bytes = workflow.source_tarball()
    # A later clock must not change the bytes (gzip headers carry a time).
    real_time = workflow.time.time
    monkeypatch.setattr("time.time", lambda: real_time() + 86400)
    assert workflow.source_tarball() == first_bytes
    first = workflow.package_source(cfg)
    second = workflow.package_source(cfg)
    assert first == second
    assert sum(k.startswith("code/") for k in s3.objects) == 1


def test_running_training_job_is_reused_not_duplicated(env):
    cfg, sm, _ = env
    name = workflow.start_training(cfg, "s3://bucket/code/x.tar.gz",
                                   {"train": "s3://bucket/data/train/", "test": "s3://bucket/data/test/"})
    again = workflow.start_training(cfg, "s3://bucket/code/x.tar.gz",
                                    {"train": "s3://bucket/data/train/", "test": "s3://bucket/data/test/"})
    assert again == name
    assert [c for c in sm.calls if c[0] == "create_training_job"] == [("create_training_job", name)]


def test_deploy_creates_then_noops_then_switches(env):
    cfg, sm, _ = env
    old = completed_job(sm, cfg, "20261006-100000")
    workflow.deploy(cfg, old)
    assert sm.endpoints[cfg.endpoint_name]["EndpointConfigName"] == "%s-epc-20261006-100000" % PREFIX

    calls = len(sm.calls)
    workflow.deploy(cfg, old)
    assert len(sm.calls) == calls, "redeploying the same model must change nothing"

    new = completed_job(sm, cfg, "20261006-110000")
    workflow.deploy(cfg, new)
    assert list(sm.endpoints) == [cfg.endpoint_name]          # still exactly one endpoint
    assert sm.endpoints[cfg.endpoint_name]["EndpointConfigName"] == "%s-epc-20261006-110000" % PREFIX
    assert set(sm.configs) == {"%s-epc-20261006-110000" % PREFIX}
    assert set(sm.models) == {"%s-model-20261006-110000" % PREFIX}


def test_deploy_refuses_an_unfinished_job(env):
    cfg, sm, _ = env
    name = completed_job(sm, cfg, "20261006-100000")
    sm.jobs[name]["TrainingJobStatus"] = "InProgress"
    with pytest.raises(workflow.DemoError, match="InProgress"):
        workflow.deploy(cfg, name)


# ------------------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------------------
def test_cleanup_order_scope_and_bucket_safety(env, monkeypatch):
    cfg, sm, s3 = env
    monkeypatch.setattr(cleanup.time, "sleep", lambda s: None)
    workflow.deploy(cfg, completed_job(sm, cfg, "20261006-100000"))
    # Lookalikes that must survive: another project, and a prefix that merely
    # starts the same way.
    sm.models["other-project-model"] = {}
    sm.models["%s2-model-20261006-100000" % PREFIX] = {}
    sm.configs["%s2-epc-x" % PREFIX] = {}
    s3.objects.update({"data/train/train.csv": b"x", "code/sourcedir-1.tar.gz": b"x",
                       "training-output/j/output/model.tar.gz": b"x",
                       "notebook-source/demo.py": b"x", "someone-elses/file.csv": b"x"})

    other = cleanup.cleanup(cfg, purge_data=True)

    deletes = [c[0] for c in sm.calls if c[0].startswith("delete")]
    assert deletes == ["delete_endpoint", "delete_endpoint_config", "delete_model"]
    assert not sm.endpoints
    assert set(sm.models) == {"other-project-model", "%s2-model-20261006-100000" % PREFIX}
    assert set(sm.configs) == {"%s2-epc-x" % PREFIX}
    # Only the demo's prefixes went; Terraform's files and the stranger stay.
    assert set(s3.objects) == {"notebook-source/demo.py", "someone-elses/file.csv"}
    assert other == ["someone-elses/file.csv"]


def test_cleanup_without_purge_keeps_data(env):
    cfg, sm, s3 = env
    workflow.deploy(cfg, completed_job(sm, cfg, "20261006-100000"))
    s3.objects["data/train/train.csv"] = b"x"
    cleanup.cleanup(cfg)
    assert not sm.endpoints and not sm.models
    assert "data/train/train.csv" in s3.objects


def test_cleanup_finds_resources_with_no_state_file(env):
    """A lost or deleted state file must not strand an endpoint."""
    cfg, sm, s3 = env
    workflow.deploy(cfg, completed_job(sm, cfg, "20261006-100000"))
    del s3.objects[workflow.STATE_KEY]
    cleanup.cleanup(cfg)
    assert not sm.endpoints and not sm.configs and not sm.models


def test_state_is_written_before_the_create_call(env):
    """If the list APIs miss a resource, the name recorded up front still finds it."""
    cfg, sm, s3 = env

    def create_then_crash(**req):
        FakeSageMaker.create_model(sm, **req)
        raise RuntimeError("process killed mid-deploy")
    sm.create_model = create_then_crash
    with pytest.raises(RuntimeError):
        workflow.deploy(cfg, completed_job(sm, cfg, "20261006-100000"))
    state = json.loads(s3.objects[workflow.STATE_KEY])
    assert state["models"] == ["%s-model-20261006-100000" % PREFIX]

    sm.list_models = lambda **_: {"Models": []}            # search finds nothing
    cleanup.cleanup(cfg)
    assert not sm.models


def test_cleanup_stops_a_running_training_job(env, monkeypatch):
    cfg, sm, _ = env
    monkeypatch.setattr(cleanup.time, "sleep", lambda s: None)
    name = completed_job(sm, cfg, "20261006-120000")
    sm.jobs[name]["TrainingJobStatus"] = "InProgress"
    cleanup.cleanup(cfg)
    assert sm.jobs[name]["TrainingJobStatus"] == "Stopped"


def test_cleanup_tolerates_a_recorded_job_that_was_never_created(env):
    """record() runs before CreateTrainingJob; that call can still fail."""
    cfg, sm, _ = env
    workflow.record(cfg, "training_jobs", "%s-train-20261006-130000" % PREFIX)
    cleanup.cleanup(cfg)          # must not raise on the missing job


class FakeRuntime:
    """Fails with the given ModelError statuses, then answers."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def invoke_endpoint(self, **_):
        self.calls += 1
        if self.statuses:
            status = self.statuses.pop(0)
            raise ClientError({"Error": {"Code": "ModelError", "Message": "x"},
                               "OriginalStatusCode": status,
                               "OriginalMessage": json.dumps({"error": "bad reading"})
                               if status == 400 else "<html>502 Bad Gateway</html>"},
                              "InvokeEndpoint")
        import io
        return {"Body": io.BytesIO(json.dumps({"predictions": []}).encode())}


def test_invoke_rides_out_a_cold_start_502(env, monkeypatch):
    """Seen live: the first call to a new serverless endpoint got nginx's 502."""
    cfg, _, _ = env
    runtime = FakeRuntime([502, 502])
    monkeypatch.setitem(workflow._CLIENTS[cfg.region], "runtime", runtime)
    body, _ = workflow.invoke(cfg, {})
    assert body == {"predictions": []} and runtime.calls == 3


def test_invoke_does_not_retry_a_400(env, monkeypatch):
    cfg, _, _ = env
    runtime = FakeRuntime([400])
    monkeypatch.setitem(workflow._CLIENTS[cfg.region], "runtime", runtime)
    with pytest.raises(workflow.DemoError, match="HTTP 400 from the model: bad reading"):
        workflow.invoke(cfg, {})
    assert runtime.calls == 1


def test_invoke_gives_up_on_a_persistent_502(env, monkeypatch):
    cfg, _, _ = env
    runtime = FakeRuntime([502] * workflow.COLD_START_ATTEMPTS)
    monkeypatch.setitem(workflow._CLIENTS[cfg.region], "runtime", runtime)
    with pytest.raises(workflow.DemoError, match="HTTP 502"):
        workflow.invoke(cfg, {})
    assert runtime.calls == workflow.COLD_START_ATTEMPTS
