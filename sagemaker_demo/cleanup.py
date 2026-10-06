"""Remove everything the Python side created, in dependency order.

Run this before `terraform destroy` (destroy.sh does). Terraform owns the
bucket, the roles and the notebook; it has never heard of the endpoint, its
config or its model, and would happily delete the role they run as while
leaving them behind.

Order matters:
  1. stop any training job still running   (the only thing here billed by the second)
  2. delete the endpoint and wait for it    (a config in use cannot be deleted)
  3. delete endpoint configs, then models
  4. delete their CloudWatch logs
  5. optionally, the demo's own S3 prefixes -- never the whole bucket

Names come from two places: the state file written before each create
call, and a search by name prefix. Either one alone is enough to find a
resource, so a crash mid-deploy or a deleted state file leaves nothing
stranded.
"""

import json
import time

from botocore.exceptions import ClientError

from . import workflow
from .workflow import DEMO_PREFIXES, STATE_KEY, log

# Terraform uploads the notebook's copy of the project here and deletes it
# itself on destroy, so cleanup leaves it alone.
TERRAFORM_PREFIXES = ("notebook-source/",)


def _paginate(call, key, **kwargs):
    items, token = [], None
    while True:
        page = call(NextToken=token, **kwargs) if token else call(**kwargs)
        items.extend(page.get(key, []))
        token = page.get("NextToken")
        if not token:
            return items


def _state(cfg):
    try:
        return workflow.load_state(cfg)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchBucket", "AccessDenied"):
            return {"training_jobs": [], "models": [], "endpoint_configs": [], "endpoints": []}
        raise


def find_resources(cfg):
    """Every demo resource that still exists, from state plus a name search."""
    sm = workflow.clients(cfg)["sagemaker"]
    state = _state(cfg)
    p = cfg.prefix

    endpoints = {e["EndpointName"] for e in _paginate(
        sm.list_endpoints, "Endpoints", NameContains=p) if e["EndpointName"] == cfg.endpoint_name}
    configs = {c["EndpointConfigName"] for c in _paginate(
        sm.list_endpoint_configs, "EndpointConfigs", NameContains="%s-epc-" % p)
        if c["EndpointConfigName"].startswith("%s-epc-" % p)}
    models = {m["ModelName"] for m in _paginate(
        sm.list_models, "Models", NameContains="%s-model-" % p)
        if m["ModelName"].startswith("%s-model-" % p)}
    running = {j["TrainingJobName"] for j in workflow.list_training_jobs(cfg, status="InProgress")}

    # Names in the state file that the search missed (a different prefix
    # match, eventual consistency) are added if they still exist.
    for name in state.get("endpoints", []):
        if workflow._exists(sm.describe_endpoint, EndpointName=name):
            endpoints.add(name)
    for name in state.get("endpoint_configs", []):
        if workflow._exists(sm.describe_endpoint_config, EndpointConfigName=name):
            configs.add(name)
    for name in state.get("models", []):
        if workflow._exists(sm.describe_model, ModelName=name):
            models.add(name)
    for name in state.get("training_jobs", []):
        job = workflow._exists(sm.describe_training_job, TrainingJobName=name)
        if job and job["TrainingJobStatus"] in ("InProgress", "Stopping"):
            running.add(name)

    return {"training_jobs": sorted(running), "endpoints": sorted(endpoints),
            "endpoint_configs": sorted(configs), "models": sorted(models)}


def _stop_training(sm, name, timeout=600):
    try:
        sm.stop_training_job(TrainingJobName=name)
    except ClientError as e:
        # Already stopping or finished between the listing and this call.
        if e.response["Error"]["Code"] != "ValidationException":
            raise
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = sm.describe_training_job(TrainingJobName=name)["TrainingJobStatus"]
        if status not in ("InProgress", "Stopping"):
            return status
        time.sleep(10)
    raise workflow.DemoError("Training job %s is still stopping after %d minutes; "
                             "run cleanup again shortly." % (name, timeout // 60))


def _wait_endpoint_deleted(sm, name, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        endpoint = workflow._exists(sm.describe_endpoint, EndpointName=name)
        if endpoint is None:
            return
        time.sleep(10)
    raise workflow.DemoError("Endpoint %s is still deleting after %d minutes; run "
                             "cleanup again before terraform destroy." % (name, timeout // 60))


def _delete_logs(cfg, logs, endpoints):
    for name in endpoints:
        group = "/aws/sagemaker/Endpoints/%s" % name
        try:
            logs.delete_log_group(logGroupName=group)
            log("deleted log group %s" % group)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
    # The training log group is shared by every job in the account and region;
    # only this demo's streams are removed.
    group = "/aws/sagemaker/TrainingJobs"
    try:
        streams = _paginate(logs.describe_log_streams, "logStreams", logGroupName=group,
                            logStreamNamePrefix="%s-train-" % cfg.prefix)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return
        raise
    for stream in streams:
        logs.delete_log_stream(logGroupName=group, logStreamName=stream["logStreamName"])
    if streams:
        log("deleted %d training log streams" % len(streams))


def bucket_contents(cfg, s3):
    """(demo-owned keys, unrelated keys) currently in the bucket."""
    ours, other = [], []
    try:
        objects = _paginate_s3(s3, cfg.bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            return [], []
        raise
    for key in objects:
        if key.startswith(DEMO_PREFIXES):
            ours.append(key)
        elif not key.startswith(TERRAFORM_PREFIXES):
            other.append(key)
    return ours, other


def _paginate_s3(s3, bucket):
    keys, token = [], None
    while True:
        kwargs = {"Bucket": bucket}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        keys.extend(o["Key"] for o in page.get("Contents", []))
        token = page.get("NextContinuationToken")
        if not token:
            return keys


def cleanup(cfg, purge_data=False, delete_logs=True):
    """Delete the Python-created resources. Returns the unrelated S3 keys left."""
    c = workflow.clients(cfg)
    sm, s3 = c["sagemaker"], c["s3"]
    found = find_resources(cfg)
    if not any(found.values()):
        log("no SageMaker training, model or endpoint resources left for prefix %s" % cfg.prefix)

    for name in found["training_jobs"]:
        log("stopping training job %s" % name)
        log("training job %s is now %s" % (name, _stop_training(sm, name)))

    for name in found["endpoints"]:
        log("deleting endpoint %s" % name)
        try:
            sm.delete_endpoint(EndpointName=name)
        except ClientError as e:
            if "Could not find" not in e.response["Error"]["Message"]:
                raise
        _wait_endpoint_deleted(sm, name)
        log("endpoint %s deleted" % name)

    for name in found["endpoint_configs"]:
        sm.delete_endpoint_config(EndpointConfigName=name)
        log("deleted endpoint config %s" % name)
    for name in found["models"]:
        sm.delete_model(ModelName=name)
        log("deleted model %s" % name)

    if delete_logs:
        _delete_logs(cfg, c["logs"], set(found["endpoints"]) | {cfg.endpoint_name})

    ours, other = bucket_contents(cfg, s3)
    if purge_data:
        for i in range(0, len(ours), 1000):
            s3.delete_objects(Bucket=cfg.bucket, Delete={
                "Objects": [{"Key": k} for k in ours[i:i + 1000]], "Quiet": True})
        if ours:
            log("deleted %d demo objects under %s" % (len(ours), ", ".join(DEMO_PREFIXES)))
    else:
        # Training job records cannot be deleted; they stay listed (free).
        try:
            state = workflow.load_state(cfg, s3)
            for kind in ("models", "endpoint_configs", "endpoints"):
                state[kind] = []
            s3.put_object(Bucket=cfg.bucket, Key=STATE_KEY,
                          Body=json.dumps(state, indent=2).encode())
        except ClientError:
            pass
        if ours:
            log("kept %d objects (data, code, artifacts) in s3://%s; add --purge-data "
                "to remove them" % (len(ours), cfg.bucket))

    if other:
        log("s3://%s also holds %d object(s) this demo did not create, e.g. %s. They "
            "were left alone; terraform destroy will refuse to delete a non-empty "
            "bucket until you move or delete them." % (cfg.bucket, len(other), other[0]))
    return other


def stop_notebook(cfg):
    """Stop (not delete) the notebook instance: billing stops, files stay."""
    name = cfg.get("notebook_name")
    if not name:
        raise workflow.DemoError("No notebook_name in the configuration.")
    sm = workflow.clients(cfg)["sagemaker"]
    status = sm.describe_notebook_instance(NotebookInstanceName=name)["NotebookInstanceStatus"]
    if status == "InService":
        sm.stop_notebook_instance(NotebookInstanceName=name)
        log("stopping notebook %s (takes a minute or two)" % name)
    else:
        log("notebook %s is %s; nothing to stop" % (name, status))
