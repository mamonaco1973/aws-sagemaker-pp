#!/usr/bin/env python3
"""Regenerate notebook/sagemaker_random_forest.ipynb from the cells below.

Edit the cells here, not in Jupyter, and run `python3 make_notebook.py`. The
notebook is written without outputs. Standard library only.
tests/test_notebook.py fails if the committed notebook and this file differ.
"""

import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "notebook", "sagemaker_random_forest.ipynb")

ARCHITECTURE = """\
```mermaid
flowchart LR
  subgraph TF["Terraform: 01-infrastructure"]
    NB["Notebook instance<br/>ml.t3.medium"]
    S3[("S3 bucket<br/>data, code, artifacts")]
  end
  subgraph PY["Python: this notebook or demo.py"]
    TJ["Training job<br/>ml.m5.large, then gone"]
    EP["Serverless endpoint<br/>model + endpoint config"]
  end
  NB -- "1 CSV + source" --> S3
  NB -- "2 CreateTrainingJob" --> TJ
  S3 -- "train / test channels" --> TJ
  TJ -- "3 model.tar.gz" --> S3
  S3 -- "4 ModelDataUrl" --> EP
  NB -- "5 InvokeEndpoint (IAM-signed)" --> EP
```"""

CELLS = [
    ("md", """\
# Random forest on SageMaker: train once, predict without retraining

This notebook trains a scikit-learn **random forest** in a managed SageMaker
training job, stores the fitted model as `model.joblib` in S3, and serves that
file from an on-demand **Serverless Inference** endpoint.

The point to watch for: **the training compute disappears when training
finishes.** What survives is one small file in S3. A separate process, in a
separate container, loads that file and keeps answering questions with no
training data and no retraining.

> **Synthetic data.** The sensor readings and labels are generated from a
> made-up formula with deliberate noise. This is a demonstration of the
> workflow, not a validated equipment failure model.

No LLM or external inference API is involved: the 200 decision trees make
every prediction themselves.

Every step has a command-line equivalent in `demo.py`, noted as **CLI** below."""),

    ("md", """\
## 1. Architecture and terminology

""" + ARCHITECTURE + """

| Term | What it is here |
|---|---|
| **Training job** | A container SageMaker starts on `ml.m5.large`, runs `ml/train.py` in, and shuts down. Billed per second while it runs. |
| **Model artifact** | `model.tar.gz` in S3: everything training wrote to `/opt/ml/model`. |
| **Model** | A SageMaker record pairing that artifact with a container image and an IAM role. No compute. |
| **Endpoint config** | How to serve a model. Here: serverless, 2 GB, at most one concurrent request. |
| **Endpoint** | The callable thing. Serverless, so no instance sits idle; you pay per request duration. |
| **Execution role** | The IAM role the training and inference containers run as. Different from this notebook's role. |

Terraform created the notebook, the bucket and both roles. **This notebook
creates the training job, model, endpoint config and endpoint**, and section
11 deletes them."""),

    ("code", """\
# Setup: find the project code and the infrastructure. Nothing is pasted in;
# demo-config.json was written by the notebook's lifecycle script from the
# Terraform outputs.
import json, os, sys
PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))

import boto3, joblib, numpy, pandas as pd, sklearn
from sagemaker_demo import config, workflow, cleanup

print("python", sys.version.split()[0], "| scikit-learn", sklearn.__version__,
      "| numpy", numpy.__version__, "| joblib", joblib.__version__)
assert sklearn.__version__ == "1.4.2", (
    "Wrong kernel: choose 'Python 3 (sagemaker-demo)'. If it is missing, the "
    "lifecycle script may still be building it; see ../kernel-setup.log")

cfg = config.load()
print("config from", cfg.source)
for key in ("region", "prefix", "bucket", "execution_role_arn", "sklearn_image_uri",
            "training_instance_type", "serverless_memory_mb"):
    print("  %-22s %s" % (key, cfg[key]))"""),

    ("code", """\
# Credentials: the default chain. On a notebook instance that is the
# notebook's own IAM role, delivered through the instance metadata service.
identity = boto3.client("sts", region_name=cfg.region).get_caller_identity()
print("calling AWS as", identity["Arn"])"""),

    ("md", """\
## 2. Generate and inspect the dataset

5,000 machines, three readings each, and a label. `ml/dataset.py` draws the
readings from fixed distributions with seed 42, computes a **failure
probability** from them, then flips a weighted coin per machine. The model
therefore sees noisy evidence, not a rule: some hot, shaky machines are
labelled healthy and some cool ones failed.

The split is **stratified** (both halves keep the same failure rate) and
happens here, before upload. The 20% test half never reaches the code that
fits the model.

**CLI:** `python demo.py train` runs sections 2 to 4 in one go."""),

    ("code", """\
paths = workflow.make_dataset()
train_df = pd.read_csv(paths["train"])
test_df = pd.read_csv(paths["test"])
print("train rows:", len(train_df), "| test rows:", len(test_df))
print("failure rate  train %.3f  test %.3f" % (
    (train_df.status == "failure").mean(), (test_df.status == "failure").mean()))
train_df.head()"""),

    ("code", """\
train_df.groupby("status").describe().T.round(2)"""),

    ("code", """\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for label, color in (("healthy", "tab:blue"), ("failure", "tab:red")):
    rows = train_df[train_df.status == label]
    axes[0].scatter(rows.temperature, rows.vibration, s=6, alpha=0.4, c=color, label=label)
axes[0].set(xlabel="temperature (C)", ylabel="vibration (mm/s)", title="Overlap: no clean boundary")
axes[0].legend()

bins = pd.cut(train_df.operating_hours, range(0, 12001, 2000))
train_df.groupby(bins, observed=True).status.apply(lambda s: (s == "failure").mean()).plot.bar(
    ax=axes[1], color="tab:gray", title="Failure rate by operating hours")
axes[1].set(xlabel="operating hours", ylabel="failure rate")
plt.tight_layout()"""),

    ("md", """\
## 3. Upload the data and the training code

Two **channels**, `train` and `test`, as separate S3 prefixes. SageMaker
copies each into the container at `/opt/ml/input/data/<channel>/`.

The training code (`ml/train.py`, `ml/inference.py`, `ml/dataset.py`) goes up
as a small tarball named by its content hash. This is SageMaker **script
mode**: the AWS scikit-learn image downloads and runs your script, so you
never build an image."""),

    ("code", """\
data_uris = workflow.upload_dataset(cfg, paths)
source_uri = workflow.package_source(cfg)
print(json.dumps(data_uris, indent=2))
print("source:", source_uri)"""),

    ("md", """\
## 4. Launch the training job and wait

Below is the complete `CreateTrainingJob` request, printed before it is
sent. Note `StoppingCondition.MaxRuntimeInSeconds`: SageMaker stops the job
there, whatever the script is doing.

Expect about **3 to 5 minutes**, mostly provisioning the instance and pulling
the image; the script itself fits 200 trees in a few seconds. If a job for
this prefix is already running, it is reused rather than duplicated.

**CLI:** `python demo.py train`, or `python demo.py wait` to reattach."""),

    ("code", """\
def show(request):
    print(json.dumps(request, indent=2, default=str))

job_name = workflow.start_training(cfg, source_uri, data_uris, print_request=show)"""),

    ("code", """\
job = workflow.wait_for_training(cfg, job_name)
summary = workflow.training_summary(cfg, job_name)
print("\\nbilled for %s seconds of %s" % (summary["billable_seconds"], summary["instance_type"]))
print("model artifact:", summary["model_artifact"])"""),

    ("md", """\
The job is **Completed**, and the `ml.m5.large` that ran it no longer
exists. You were billed for the seconds above and nothing more. Everything
that is left of training is the artifact in S3.

## 5. Evaluation results

`train.py` scored the model on the 1,000 held-out rows once, after fitting,
and printed the results. SageMaker scraped them from the log into the
training job's record (`FinalMetricDataList`).

* **Precision**: of the machines it flagged as failing, the share that did.
* **Recall**: of the machines that failed, the share it caught.

**CLI:** `python demo.py evaluate`"""),

    ("code", """\
m = summary["metrics"]
print("accuracy  %.3f\\nprecision %.3f\\nrecall    %.3f" % (
    m["test:accuracy"], m["test:precision"], m["test:recall"]))
pd.DataFrame(
    [[int(m["test:true_healthy"]), int(m["test:false_failure"])],
     [int(m["test:missed_failure"]), int(m["test:true_failure"])]],
    index=["actual healthy", "actual failure"],
    columns=["predicted healthy", "predicted failure"])"""),

    ("md", """\
Reasonable but far from perfect, as it should be: the labels were drawn with
noise, so no model can recover them exactly. Missed failures (bottom left)
are the costly kind in real maintenance; a real project would tune the
decision threshold on a validation split. This demo deliberately does no
tuning, because tuning against the test set would make its score a lie.

## 6. Locate and download the model artifact

The training job's `ModelArtifacts.S3ModelArtifacts` points at the archive.
It sits under `training-output/<job>/output/`.

**CLI:** `python demo.py artifact`"""),

    ("code", """\
s3 = boto3.client("s3", region_name=cfg.region)
listing = s3.list_objects_v2(Bucket=cfg.bucket, Prefix="training-output/%s/" % job_name)
for obj in listing.get("Contents", []):
    print("%10d bytes  s3://%s/%s" % (obj["Size"], cfg.bucket, obj["Key"]))

archive, model_dir = workflow.download_artifact(cfg, job_name)
print("\\ndownloaded to", os.path.relpath(archive))"""),

    ("md", """\
## 7. Inspect the archive and the trained model

| File | Purpose |
|---|---|
| `model.joblib` | The fitted `RandomForestClassifier`: 200 trees of split thresholds and leaf class counts, plus fitted attributes such as `classes_` and `feature_names_in_`. |
| `metadata.json` | Feature order, class labels, hyperparameters and the library versions it was trained with. |
| `evaluation.json` | The held-out metrics from section 5. |
| `samples.json` | Three new readings the model scored low, borderline and high (section 10). |
| `code/inference.py` | The serving code. It travels inside the artifact. |

`model.joblib` is a **pickle of scikit-learn objects**, not a portable model
format. Loading it needs a compatible Python and the same scikit-learn
(1.4.2 here). That is why the notebook kernel pins the versions the training
container used, and why the endpoint uses the same container image."""),

    ("code", """\
for name, size in workflow.archive_listing(archive):
    print("%9d  %s" % (size, name))
with open(os.path.join(model_dir, "metadata.json")) as f:
    metadata = json.load(f)
print(json.dumps({k: metadata[k] for k in ("feature_order", "classes", "hyperparameters", "runtime")}, indent=2))"""),

    ("code", """\
model = joblib.load(os.path.join(model_dir, "model.joblib"))
nodes = [t.tree_.node_count for t in model.estimators_]
print(type(model).__name__, "with", len(model.estimators_), "trees")
print("classes_          ", list(model.classes_))
print("feature_names_in_ ", list(model.feature_names_in_))
print("nodes per tree     min %d, mean %d, max %d" % (min(nodes), sum(nodes) / len(nodes), max(nodes)))
print("deepest tree       %d levels" % max(t.get_depth() for t in model.estimators_))
pd.Series(model.feature_importances_, index=model.feature_names_in_).round(3)"""),

    ("md", """\
## 8. Look inside one tree

Each prediction is a vote: every tree routes a reading down its splits to a
leaf, and `predict_proba` averages the leaves' failure shares across all 200
trees. Here are the top three levels of the first tree, with the training
rows that reached each node. The real tree continues up to eight levels."""),

    ("code", """\
from sklearn.tree import plot_tree, export_text

fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(model.estimators_[0], max_depth=2, feature_names=list(model.feature_names_in_),
          class_names=list(model.classes_), filled=True, impurity=False, proportion=True,
          rounded=True, fontsize=9, ax=ax)
plt.show()
print(export_text(model.estimators_[0], feature_names=list(model.feature_names_in_), max_depth=2))"""),

    ("md", """\
## 9. Deploy to Serverless Inference

Three API calls, all printed below:

1. **CreateModel** pairs the artifact with the same scikit-learn image that
   trained it, and the execution role.
2. **CreateEndpointConfig** asks for `ServerlessConfig` (2 GB, max
   concurrency 1) and no provisioned concurrency, so nothing runs between
   requests.
3. **CreateEndpoint** brings it up: about **3 to 6 minutes**.

When a request arrives, SageMaker starts the container and extracts
`model.tar.gz` into **`/opt/ml/model/`**. The container imports
`/opt/ml/model/code/inference.py` and calls its `model_fn`, which runs the
one and only `joblib.load()` in this process:"""),

    ("code", """\
import inspect, inference
print(inspect.getsource(inference.model_fn))"""),

    ("md", """\
Running the deploy cell again is safe. An endpoint that already serves this
model is left alone; one serving an older model is switched over, and the
old model and config are deleted.

**CLI:** `python demo.py deploy`"""),

    ("code", """\
endpoint = workflow.deploy(cfg, job_name, print_request=show)
print(endpoint["EndpointName"], endpoint["EndpointStatus"])"""),

    ("md", """\
## 10. Send new readings and get predictions

`samples.json` holds three readings the model has **never seen**. Training
scored 200 fresh readings and kept the lowest-risk, the closest to 50/50, and
the highest-risk, along with the probability it gave each. So the expected
answers come from the model itself, not from a claim in this notebook.

`InvokeEndpoint` is an AWS API call signed with this notebook's IAM
credentials. The endpoint has no public URL; a caller needs
`sagemaker:InvokeEndpoint` on it.

The **first request may take several seconds**: a serverless endpoint that
has been idle has no container running, so the first call pays for starting
one and loading the model (a *cold start*). The calls after it reuse the
warm container.

**CLI:** `python demo.py predict`"""),

    ("code", """\
import time
samples = workflow.load_samples(model_dir)
local = model.predict_proba(pd.DataFrame([s["reading"] for s in samples]))[:, list(model.classes_).index("failure")]

rows = []
for sample, local_p in zip(samples, local):
    body, seconds = workflow.invoke(cfg, sample["reading"])
    result = body["predictions"][0]
    rows.append({**sample["reading"], "endpoint label": result["label"],
                 "endpoint p(failure)": result["failure_probability"],
                 "local p(failure)": round(float(local_p), 4), "seconds": round(seconds, 2)})
pd.DataFrame(rows, index=[s["name"] for s in samples])"""),

    ("md", """\
The endpoint and the copy loaded in this notebook give the same answers:
same file, same trees. The `seconds` column usually shows the cold start on
the first row only.

Several readings fit in one request. The field order in the request does not
matter: `inference.py` maps the names onto the training column order."""),

    ("code", """\
batch = {"instances": [
    {"operating_hours": 1200, "vibration": 1.9, "temperature": 62.0},
    {"vibration": 4.8, "temperature": 84.5, "operating_hours": 10400},
]}
body, seconds = workflow.invoke(cfg, batch)
print(json.dumps(body, indent=2), "\\n%.2fs" % seconds)"""),

    ("md", """\
Bad requests come back as **HTTP 400** with a reason, not as a guess.
`inference.py` checks for missing and unknown fields, non-numbers and
non-finite values before the model sees anything."""),

    ("code", """\
bad_requests = {
    "missing field": {"temperature": 70, "vibration": 3.0},
    "typo in a name": {"temprature": 70, "vibration": 3.0, "operating_hours": 100},
    "string value": {"temperature": "hot", "vibration": 3.0, "operating_hours": 100},
    "infinity": {"temperature": float("inf"), "vibration": 3.0, "operating_hours": 100},
}
for what, payload in bad_requests.items():
    try:
        workflow.invoke(cfg, payload)
        print("%-15s accepted (unexpected)" % what)
    except workflow.DemoError as e:
        print("%-15s %s" % (what, e))"""),

    ("md", """\
The endpoint's own log shows the load happening once per serving process,
and nowhere else (CloudWatch can take a minute to catch up):"""),

    ("code", """\
for line in workflow.endpoint_log_lines(cfg, "joblib.load"):
    print(line)"""),

    ("md", """\
## 11. Clean up

`cleanup.cleanup()` deletes what this notebook created, in dependency order:
any running training job is stopped, then the endpoint (waiting until it is
gone), the endpoint configs, the models, and their CloudWatch logs. It finds
them by the names recorded in `s3://<bucket>/state/resources.json` and by
name prefix, so it works even after a crash. Training job records cannot be
deleted; they stay listed and cost nothing.

The data and artifacts in S3 are kept; `./destroy.sh` removes them.

**CLI:** `python demo.py cleanup`"""),

    ("code", """\
cleanup.cleanup(cfg)
print(json.dumps(cleanup.find_resources(cfg), indent=2))"""),

    ("md", """\
### The notebook is still running

This notebook instance is Terraform's and costs about $0.05 an hour while
it is **InService**. When you are done:

* **Pause:** stop it from the SageMaker console (Notebook instances, then
  Stop) or run `python demo.py notebook-stop`. Files under
  `/home/ec2-user/SageMaker` are kept.
* **Remove everything:** run `./destroy.sh` on the machine you deployed
  from. It runs this cleanup again with `--purge-data`, then
  `terraform destroy`."""),
]


def cell(kind, source):
    lines = source.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def main(out=OUT):
    notebook = {
        "cells": [cell(kind, source) for kind, source in CELLS],
        "metadata": {
            "kernelspec": {"name": "sagemaker-demo", "display_name": "Python 3 (sagemaker-demo)",
                           "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(notebook["cells"]):
        c["id"] = "cell-%02d" % i
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(notebook, f, indent=1)
        f.write("\n")
    print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    # An explicit path is for the test that checks the committed notebook.
    main(*sys.argv[1:2])
