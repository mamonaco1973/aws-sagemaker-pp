"""Train, inspect, deploy and invoke -- plain boto3 calls, one step each.

The SageMaker Python SDK is deliberately not used. Each step below is one or
two API requests, and the notebook prints the request before sending it, so
nothing about what SageMaker is asked to do is hidden behind a helper.

Resources this module creates (Terraform does not know about them):

    training job     <prefix>-train-<UTC timestamp>   record only; cannot be deleted
    model            <prefix>-model-<same timestamp>
    endpoint config  <prefix>-epc-<same timestamp>
    endpoint         <prefix>-endpoint                one, reused across runs

Every name is written to s3://<bucket>/state/resources.json *before* the
create call, so cleanup can find it even if the call or the process dies
halfway. Cleanup also searches by name prefix as a second net.
"""

import gzip
import hashlib
import importlib
import io
import json
import os
import sys
import tarfile
import time

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import PROJECT_ROOT

ML_DIR = os.path.join(PROJECT_ROOT, "ml")

SOURCE_FILES = ("train.py", "inference.py", "dataset.py")
STATE_KEY = "state/resources.json"
DATA_PREFIX = "data/"
CODE_PREFIX = "code/"
OUTPUT_PREFIX = "training-output/"
# Everything the Python side writes to the bucket. Cleanup with --purge-data
# deletes these prefixes and nothing else.
DEMO_PREFIXES = (DATA_PREFIX, CODE_PREFIX, OUTPUT_PREFIX, "state/")

HYPERPARAMETERS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10,
                   "random_state": 42}

# train.py prints "METRIC accuracy=0.8540"; SageMaker scrapes the training
# log with these and reports the values in DescribeTrainingJob. The last four
# are the confusion matrix cells (actual -> predicted):
#   true_healthy   healthy -> healthy      false_failure  healthy -> failure
#   missed_failure failure -> healthy      true_failure   failure -> failure
METRIC_DEFINITIONS = [
    {"Name": "test:%s" % name, "Regex": r"METRIC %s=([0-9.]+)" % name}
    for name in ("accuracy", "precision", "recall", "true_healthy", "false_failure",
                 "missed_failure", "true_failure")
]

TRAINING_TIMEOUT_SECONDS = 30 * 60   # provisioning + image pull + max runtime
ENDPOINT_TIMEOUT_SECONDS = 20 * 60
POLL_SECONDS = 15
COLD_START_ATTEMPTS = 6
COLD_START_RETRY_SECONDS = 5


class DemoError(RuntimeError):
    """A failure with a message that says what to do next."""


def log(message):
    print("NOTE: %s" % message, flush=True)


# ------------------------------------------------------------------------------
# Clients and state
# ------------------------------------------------------------------------------
def session(cfg):
    return boto3.session.Session(region_name=cfg.region)


_CLIENTS = {}


def clients(cfg):
    """boto3 clients for the configured region, created once and reused."""
    if cfg.region not in _CLIENTS:
        s = session(cfg)
        # A serverless cold start can take tens of seconds; the default 60s read
        # timeout is just enough, so give the runtime client some headroom.
        runtime_cfg = BotoConfig(read_timeout=90, retries={"max_attempts": 3, "mode": "standard"})
        _CLIENTS[cfg.region] = {
            "sagemaker": s.client("sagemaker"),
            "runtime": s.client("sagemaker-runtime", config=runtime_cfg),
            "s3": s.client("s3"),
            "logs": s.client("logs"),
        }
    return _CLIENTS[cfg.region]


def tags(cfg):
    return [{"Key": "Project", "Value": cfg.get("project", "aws-sagemaker-pp")},
            {"Key": "Prefix", "Value": cfg.prefix},
            {"Key": "ManagedBy", "Value": "python"}]


def load_state(cfg, s3=None):
    s3 = s3 or clients(cfg)["s3"]
    try:
        body = s3.get_object(Bucket=cfg.bucket, Key=STATE_KEY)["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"training_jobs": [], "models": [], "endpoint_configs": [], "endpoints": []}
        raise
    return json.loads(body)


def record(cfg, kind, name, s3=None):
    """Remember a resource name before creating it."""
    s3 = s3 or clients(cfg)["s3"]
    state = load_state(cfg, s3)
    if name not in state[kind]:
        state[kind].append(name)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        s3.put_object(Bucket=cfg.bucket, Key=STATE_KEY,
                      Body=json.dumps(state, indent=2).encode(),
                      ContentType="application/json")


def forget(cfg, kind, name, s3=None):
    s3 = s3 or clients(cfg)["s3"]
    state = load_state(cfg, s3)
    if name in state[kind]:
        state[kind].remove(name)
        s3.put_object(Bucket=cfg.bucket, Key=STATE_KEY,
                      Body=json.dumps(state, indent=2).encode(),
                      ContentType="application/json")


# ------------------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------------------
def ml_module(name):
    """Import a file from ml/ -- the same code the container runs.

    Imported on demand so status and cleanup need only boto3, not numpy.
    """
    if ML_DIR not in sys.path:
        sys.path.insert(0, ML_DIR)
    return importlib.import_module(name)


def make_dataset(local_dir=None):
    """Generate, split and write train.csv / test.csv. Same rows every run."""
    dataset = ml_module("dataset")
    local_dir = local_dir or os.path.join(PROJECT_ROOT, "data")
    features, labels = dataset.generate()
    x_train, x_test, y_train, y_test = dataset.split(features, labels)
    paths = {}
    for channel, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
        os.makedirs(os.path.join(local_dir, channel), exist_ok=True)
        path = os.path.join(local_dir, channel, "%s.csv" % channel)
        with open(path, "w") as f:
            f.write(dataset.to_csv(x, y))
        paths[channel] = path
    return paths


def upload_dataset(cfg, paths):
    """Upload the CSVs. Same keys every run, so reruns overwrite, not pile up."""
    s3 = clients(cfg)["s3"]
    uris = {}
    for channel, path in paths.items():
        key = "%s%s/%s" % (DATA_PREFIX, channel, os.path.basename(path))
        s3.upload_file(path, cfg.bucket, key)
        uris[channel] = "s3://%s/%s%s/" % (cfg.bucket, DATA_PREFIX, channel)
        log("uploaded %s -> s3://%s/%s" % (os.path.relpath(path, PROJECT_ROOT), cfg.bucket, key))
    return uris


def source_tarball():
    """ml/*.py as a gzipped tar, byte-identical for identical files."""
    buffer = io.BytesIO()
    # gzip stamps the current time into its header unless told otherwise,
    # which would give the same files a new hash every second.
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in SOURCE_FILES:
            with open(os.path.join(ML_DIR, name), "rb") as f:
                data = f.read()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0          # reproducible: same files, same hash
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def package_source(cfg):
    """Upload ml/*.py as the tarball the container downloads and runs.

    Named by content hash: an unchanged source is not re-uploaded, and a
    training job always points at exactly the code it ran.
    """
    blob = source_tarball()
    key = "%ssourcedir-%s.tar.gz" % (CODE_PREFIX, hashlib.sha256(blob).hexdigest()[:12])
    s3 = clients(cfg)["s3"]
    try:
        s3.head_object(Bucket=cfg.bucket, Key=key)
    except ClientError:
        s3.put_object(Bucket=cfg.bucket, Key=key, Body=blob)
        log("uploaded training source -> s3://%s/%s" % (cfg.bucket, key))
    return "s3://%s/%s" % (cfg.bucket, key)


# ------------------------------------------------------------------------------
# 2. Training
# ------------------------------------------------------------------------------
def training_job_request(cfg, job_name, source_uri, data_uris):
    """The complete CreateTrainingJob request. Printed by the notebook."""
    # Script mode: the container downloads sagemaker_submit_directory and runs
    # sagemaker_program with every other hyperparameter as a --flag. Values
    # are strings; the two sagemaker_* paths are JSON strings, as the SDK
    # sends them.
    hyperparameters = {
        "sagemaker_program": json.dumps("train.py"),
        "sagemaker_submit_directory": json.dumps(source_uri),
        "sagemaker_region": json.dumps(cfg.region),
        "sagemaker_container_log_level": "20",
    }
    hyperparameters.update({k: str(v) for k, v in HYPERPARAMETERS.items()})

    def channel(name):
        return {
            "ChannelName": name,
            "ContentType": "text/csv",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": data_uris[name],
                "S3DataDistributionType": "FullyReplicated",
            }},
        }

    return {
        "TrainingJobName": job_name,
        "RoleArn": cfg.execution_role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": cfg.sklearn_image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": METRIC_DEFINITIONS,
        },
        "HyperParameters": hyperparameters,
        "InputDataConfig": [channel("train"), channel("test")],
        "OutputDataConfig": {"S3OutputPath": "s3://%s/%s" % (cfg.bucket, OUTPUT_PREFIX)},
        "ResourceConfig": {
            "InstanceType": cfg.training_instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 5,
        },
        # The hard cap: SageMaker stops the job (and the bill) at this point
        # even if the script hangs.
        "StoppingCondition": {"MaxRuntimeInSeconds": int(cfg.training_max_runtime_seconds)},
        # Debugger's system profiler is not needed and writes extra S3 data.
        "ProfilerConfig": {"DisableProfiler": True},
        "Tags": tags(cfg),
    }


def list_training_jobs(cfg, status=None):
    sm = clients(cfg)["sagemaker"]
    kwargs = {"NameContains": "%s-train-" % cfg.prefix, "SortBy": "CreationTime",
              "SortOrder": "Descending", "MaxResults": 50}
    if status:
        kwargs["StatusEquals"] = status
    jobs = sm.list_training_jobs(**kwargs)["TrainingJobSummaries"]
    return [j for j in jobs if j["TrainingJobName"].startswith("%s-train-" % cfg.prefix)]


def latest_completed_job(cfg):
    jobs = list_training_jobs(cfg, status="Completed")
    if not jobs:
        raise DemoError("No completed training job for prefix %s. Run the training "
                        "step first (python demo.py train)." % cfg.prefix)
    return jobs[0]["TrainingJobName"]


def start_training(cfg, source_uri, data_uris, print_request=None):
    """Start a training job, unless one is already running for this prefix.

    A second click on "run" while a job is in progress waits on that job
    instead of paying for a duplicate.
    """
    running = list_training_jobs(cfg, status="InProgress")
    if running:
        name = running[0]["TrainingJobName"]
        log("training job %s is already in progress; reusing it" % name)
        return name

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    name = "%s-train-%s" % (cfg.prefix, stamp)
    request = training_job_request(cfg, name, source_uri, data_uris)
    if print_request:
        print_request(request)
    record(cfg, "training_jobs", name)
    clients(cfg)["sagemaker"].create_training_job(**request)
    log("started training job %s" % name)
    return name


def wait_for_training(cfg, name, timeout=TRAINING_TIMEOUT_SECONDS):
    """Poll until the job ends. Prints each status change once."""
    sm = clients(cfg)["sagemaker"]
    deadline = time.time() + timeout
    last = None
    while True:
        job = sm.describe_training_job(TrainingJobName=name)
        status, secondary = job["TrainingJobStatus"], job.get("SecondaryStatus")
        if (status, secondary) != last:
            log("%s: %s / %s" % (name, status, secondary))
            last = (status, secondary)
        if status == "Completed":
            return job
        if status in ("Failed", "Stopped"):
            raise DemoError(
                "Training job %s %s: %s\nLogs: CloudWatch log group "
                "/aws/sagemaker/TrainingJobs, stream prefix %s"
                % (name, status.lower(), job.get("FailureReason", "no reason given"), name))
        if time.time() > deadline:
            raise DemoError(
                "Training job %s is still %s after %d minutes. It keeps running (and is "
                "capped by MaxRuntimeInSeconds); wait again with: python demo.py wait %s"
                % (name, secondary, timeout // 60, name))
        time.sleep(POLL_SECONDS)


def training_summary(cfg, name):
    job = clients(cfg)["sagemaker"].describe_training_job(TrainingJobName=name)
    return {
        "name": name,
        "status": job["TrainingJobStatus"],
        "instance_type": job["ResourceConfig"]["InstanceType"],
        "training_seconds": job.get("TrainingTimeInSeconds"),
        "billable_seconds": job.get("BillableTimeInSeconds"),
        "metrics": {m["MetricName"]: round(m["Value"], 4)
                    for m in job.get("FinalMetricDataList", [])},
        "model_artifact": job.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
        "started": str(job.get("TrainingStartTime")),
        "ended": str(job.get("TrainingEndTime")),
    }


# ------------------------------------------------------------------------------
# 3. The artifact
# ------------------------------------------------------------------------------
def split_s3_uri(uri):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def download_artifact(cfg, name, dest_root=None):
    """Download model.tar.gz and extract it. Returns (archive path, directory)."""
    summary = training_summary(cfg, name)
    uri = summary["model_artifact"]
    if not uri:
        raise DemoError("Training job %s has no model artifact (status %s)."
                        % (name, summary["status"]))
    dest = os.path.join(dest_root or os.path.join(PROJECT_ROOT, "artifacts"), name)
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, "model.tar.gz")
    bucket, key = split_s3_uri(uri)
    clients(cfg)["s3"].download_file(bucket, key, archive)
    extracted = os.path.join(dest, "model")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(extracted, filter="data")
        else:  # Python without the tarfile filter backport: refuse odd paths
            for member in tar.getmembers():
                if member.name.startswith(("/", "..")) or ".." in member.name.split("/"):
                    raise DemoError("Unsafe path in archive: %s" % member.name)
            tar.extractall(extracted)
    return archive, extracted


def archive_listing(archive):
    with tarfile.open(archive) as tar:
        return [(m.name, m.size) for m in tar.getmembers() if m.isfile()]


# ------------------------------------------------------------------------------
# 4. Serverless deployment
# ------------------------------------------------------------------------------
def _suffix(job_name, cfg):
    return job_name[len("%s-train-" % cfg.prefix):]


def model_name_for(cfg, job_name):
    return "%s-model-%s" % (cfg.prefix, _suffix(job_name, cfg))


def endpoint_config_name_for(cfg, job_name):
    return "%s-epc-%s" % (cfg.prefix, _suffix(job_name, cfg))


def model_request(cfg, job_name, artifact_uri):
    return {
        "ModelName": model_name_for(cfg, job_name),
        "ExecutionRoleArn": cfg.execution_role_arn,
        "PrimaryContainer": {
            # The same image the training job ran, so the scikit-learn that
            # unpickles model.joblib is the one that pickled it.
            "Image": cfg.sklearn_image_uri,
            "ModelDataUrl": artifact_uri,
            "Environment": {
                # The serving code ships inside model.tar.gz (train.py put it
                # in code/), so the endpoint needs nothing else from S3.
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_REGION": cfg.region,
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                # One worker process: one copy of the forest in memory.
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                # At start-up the container pip-installs the code directory.
                # With build isolation on, pip first downloads setuptools from
                # PyPI; off, it uses the copy in the image, so the endpoint
                # never needs the internet to start. ("false" really does mean
                # off: pip reads PIP_NO_* variables inverted.)
                "PIP_NO_BUILD_ISOLATION": "false",
                # The image's Python is the distribution's (PEP 668), so that
                # same pip install is refused as "externally-managed-environment"
                # unless allowed. The image's own pip.conf allows it only for
                # root's HOME, which the endpoint did not use.
                "PIP_BREAK_SYSTEM_PACKAGES": "1",
                # The server runs as a non-root user, so that install lands in
                # ~/.local, which is not on the running process's sys.path.
                # Putting the artifact's code directory on the path makes the
                # import independent of where pip put the package.
                "PYTHONPATH": "/opt/ml/model/code",
            },
        },
        "Tags": tags(cfg),
    }


def endpoint_config_request(cfg, job_name):
    return {
        "EndpointConfigName": endpoint_config_name_for(cfg, job_name),
        "ProductionVariants": [{
            "VariantName": "AllTraffic",
            "ModelName": model_name_for(cfg, job_name),
            # Serverless: no instance to leave running. No
            # ProvisionedConcurrency, so it scales to zero between calls.
            "ServerlessConfig": {
                "MemorySizeInMB": int(cfg.serverless_memory_mb),
                "MaxConcurrency": int(cfg.serverless_max_concurrency),
            },
        }],
        "Tags": tags(cfg),
    }


def _exists(call, **kwargs):
    """The Describe* result, or None if the resource does not exist."""
    try:
        return call(**kwargs)
    except ClientError as e:
        error = e.response["Error"]
        message = error.get("Message", "").lower()
        # DescribeModel/EndpointConfig/Endpoint say "Could not find ...";
        # DescribeTrainingJob says "Requested resource not found."
        if error["Code"] == "ResourceNotFound" or (
                error["Code"] == "ValidationException"
                and ("could not find" in message or "not found" in message)):
            return None
        raise


def describe_endpoint(cfg, sm=None):
    sm = sm or clients(cfg)["sagemaker"]
    return _exists(sm.describe_endpoint, EndpointName=cfg.endpoint_name)


def wait_for_endpoint(cfg, timeout=ENDPOINT_TIMEOUT_SECONDS):
    sm = clients(cfg)["sagemaker"]
    deadline = time.time() + timeout
    last = None
    while True:
        endpoint = describe_endpoint(cfg, sm)
        if endpoint is None:
            raise DemoError("Endpoint %s disappeared while waiting." % cfg.endpoint_name)
        status = endpoint["EndpointStatus"]
        if status != last:
            log("%s: %s" % (cfg.endpoint_name, status))
            last = status
        if status == "InService":
            return endpoint
        if status == "Failed":
            raise DemoError(
                "Endpoint %s failed: %s\nLogs: CloudWatch log group %s. Then remove it "
                "with: python demo.py cleanup"
                % (cfg.endpoint_name, endpoint.get("FailureReason"), cfg.endpoint_log_group))
        if time.time() > deadline:
            raise DemoError("Endpoint %s is still %s after %d minutes; check again with "
                            "python demo.py status" % (cfg.endpoint_name, status, timeout // 60))
        time.sleep(POLL_SECONDS)


def deploy(cfg, job_name, print_request=None):
    """Create or update the one serverless endpoint to serve job_name's model.

    Safe to repeat: an endpoint already serving this model is left alone, an
    endpoint serving an older model is switched over (then the old model and
    config are deleted), and a failed endpoint is replaced.
    """
    sm = clients(cfg)["sagemaker"]
    summary = training_summary(cfg, job_name)
    if summary["status"] != "Completed":
        raise DemoError("Training job %s is %s; deploy needs a completed job."
                        % (job_name, summary["status"]))

    model_req = model_request(cfg, job_name, summary["model_artifact"])
    config_req = endpoint_config_request(cfg, job_name)
    if print_request:
        print_request(model_req)
        print_request(config_req)
        print_request({"EndpointName": cfg.endpoint_name,
                       "EndpointConfigName": config_req["EndpointConfigName"]})

    if _exists(sm.describe_model, ModelName=model_req["ModelName"]) is None:
        record(cfg, "models", model_req["ModelName"])
        sm.create_model(**model_req)
        log("created model %s" % model_req["ModelName"])
    if _exists(sm.describe_endpoint_config,
               EndpointConfigName=config_req["EndpointConfigName"]) is None:
        record(cfg, "endpoint_configs", config_req["EndpointConfigName"])
        sm.create_endpoint_config(**config_req)
        log("created endpoint config %s" % config_req["EndpointConfigName"])

    endpoint = describe_endpoint(cfg, sm)
    if endpoint and endpoint["EndpointStatus"] in ("Creating", "Updating", "SystemUpdating"):
        log("endpoint is %s; waiting for it to settle first" % endpoint["EndpointStatus"])
        try:
            endpoint = wait_for_endpoint(cfg)
        except DemoError:
            endpoint = describe_endpoint(cfg, sm)
    if endpoint and endpoint["EndpointStatus"] == "Failed":
        log("deleting failed endpoint %s before recreating it" % cfg.endpoint_name)
        sm.delete_endpoint(EndpointName=cfg.endpoint_name)
        _wait_until_gone(cfg, sm)
        endpoint = None

    previous = None
    if endpoint is None:
        record(cfg, "endpoints", cfg.endpoint_name)
        sm.create_endpoint(EndpointName=cfg.endpoint_name,
                           EndpointConfigName=config_req["EndpointConfigName"],
                           Tags=tags(cfg))
        log("creating endpoint %s (serverless, typically 3-6 minutes)" % cfg.endpoint_name)
    elif endpoint["EndpointConfigName"] != config_req["EndpointConfigName"]:
        previous = endpoint["EndpointConfigName"]
        sm.update_endpoint(EndpointName=cfg.endpoint_name,
                           EndpointConfigName=config_req["EndpointConfigName"])
        log("switching endpoint %s from %s to %s"
            % (cfg.endpoint_name, previous, config_req["EndpointConfigName"]))
    else:
        log("endpoint %s already serves %s" % (cfg.endpoint_name, model_req["ModelName"]))

    endpoint = wait_for_endpoint(cfg)
    if previous:
        _delete_config_and_model(cfg, sm, previous)
    return endpoint


def _wait_until_gone(cfg, sm, timeout=ENDPOINT_TIMEOUT_SECONDS):
    deadline = time.time() + timeout
    while describe_endpoint(cfg, sm) is not None:
        if time.time() > deadline:
            raise DemoError("Endpoint %s is still deleting after %d minutes."
                            % (cfg.endpoint_name, timeout // 60))
        time.sleep(POLL_SECONDS)


def _delete_config_and_model(cfg, sm, config_name):
    config = _exists(sm.describe_endpoint_config, EndpointConfigName=config_name)
    if config is None:
        return
    sm.delete_endpoint_config(EndpointConfigName=config_name)
    forget(cfg, "endpoint_configs", config_name)
    for variant in config["ProductionVariants"]:
        if _exists(sm.describe_model, ModelName=variant["ModelName"]) is not None:
            sm.delete_model(ModelName=variant["ModelName"])
            forget(cfg, "models", variant["ModelName"])
    log("deleted superseded %s and its model" % config_name)


# ------------------------------------------------------------------------------
# 5. Invocation
# ------------------------------------------------------------------------------
def invoke(cfg, payload):
    """POST JSON to the endpoint through SageMaker Runtime.

    There is no public URL: InvokeEndpoint is an AWS API call, signed with
    the caller's IAM credentials and allowed by sagemaker:InvokeEndpoint.
    Returns (parsed response, seconds taken).
    """
    runtime = clients(cfg)["runtime"]
    started = time.time()
    for attempt in range(1, COLD_START_ATTEMPTS + 1):
        try:
            response = runtime.invoke_endpoint(
                EndpointName=cfg.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode(),
            )
            break
        except ClientError as e:
            # On a cold start the container's nginx can answer before the
            # Python server behind it is up, and the request comes back as
            # a 502/503/504 ModelError. Seen on the first call to a new
            # endpoint; the retry a few seconds later succeeds.
            # Also status 0, "could not get a response", seen live on a cold
            # endpoint through the MCP connector.
            status = e.response.get("OriginalStatusCode")
            if (e.response["Error"]["Code"] == "ModelError" and status in (0, None, 502, 503, 504)
                    and attempt < COLD_START_ATTEMPTS):
                log("HTTP %s while the endpoint starts (cold start); retrying in %ds"
                    % (status, COLD_START_RETRY_SECONDS))
                time.sleep(COLD_START_RETRY_SECONDS)
                continue
            _raise_invoke_error(cfg, e)
    body = json.loads(response["Body"].read())
    return body, time.time() - started


def _raise_invoke_error(cfg, e):
    error = e.response["Error"]
    # A 400 from inference.py arrives as ModelError, with the container's
    # own response body in OriginalMessage.
    if error["Code"] == "ModelError":
        raise DemoError("HTTP %s from the model: %s" % (
            e.response.get("OriginalStatusCode", "?"), _model_error_text(e.response)))
    # SageMaker Runtime says ValidationError here, not ValidationException.
    if error["Code"].startswith("Validation") and "not found" in error["Message"].lower():
        raise DemoError("Endpoint %s does not exist. Deploy first: python demo.py deploy"
                        % cfg.endpoint_name)
    raise e


def _model_error_text(response):
    original = response.get("OriginalMessage") or response["Error"]["Message"]
    try:
        return json.loads(original)["error"]
    except (ValueError, KeyError, TypeError):
        return original


def endpoint_log_lines(cfg, pattern, limit=20):
    """Lines from the endpoint's CloudWatch logs that contain pattern."""
    logs = clients(cfg)["logs"]
    try:
        events = logs.filter_log_events(logGroupName=cfg.endpoint_log_group,
                                        filterPattern='"%s"' % pattern,
                                        limit=limit)["events"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return []
        raise
    return [e["message"].rstrip() for e in events]


def load_samples(extracted_dir):
    with open(os.path.join(extracted_dir, "samples.json")) as f:
        return json.load(f)
