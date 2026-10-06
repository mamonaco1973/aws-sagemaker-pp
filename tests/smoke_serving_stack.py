"""Drive inference.py through the scikit-learn container's own serving code.

Not a pytest module: it needs an image with the container's serving stack
(sagemaker-sklearn-container v1.4-2-py312 and its pinned Flask, Werkzeug,
protobuf and sagemaker-containers), which tests/run_local_checks.sh builds.

What it reproduces, with no AWS involved:
  - model.tar.gz extracted at /opt/ml/model, as SageMaker does
  - the environment workflow.model_request() sets on the SageMaker model
  - the container's WSGI app answering POST /invocations, which imports
    /opt/ml/model/code/inference.py exactly as the endpoint will
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def prepare():
    """Train locally and lay the artifact out exactly as SageMaker would."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from conftest import run_training

    # Train, pack model.tar.gz as SageMaker does, extract it to /opt/ml/model.
    work = tempfile.mkdtemp()
    model_dir = run_training(work)
    archive = os.path.join(work, "model.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for name in os.listdir(model_dir):
            tar.add(os.path.join(model_dir, name), arcname=name)
    shutil.rmtree("/opt/ml/model", ignore_errors=True)
    os.makedirs("/opt/ml/model")
    with tarfile.open(archive) as tar:
        tar.extractall("/opt/ml/model", filter="data")
    shutil.rmtree("/opt/ml/code", ignore_errors=True)
    shutil.rmtree("/tmp/model-server", ignore_errors=True)


def model_environment():
    """The Environment the SageMaker model sets, from workflow.model_request()."""
    from sagemaker_demo import workflow
    from sagemaker_demo.config import Config

    cfg = Config(region="us-east-1", prefix="smoke", execution_role_arn="x",
                 sklearn_image_uri="x", training_instance_type="x",
                 training_max_runtime_seconds=1, serverless_memory_mb=2048,
                 serverless_max_concurrency=1, bucket="x")
    return workflow.model_request(cfg, "smoke-train-1", "s3://x")["PrimaryContainer"]["Environment"]


def serve():
    """A fresh process started with the model's environment, as SageMaker
    starts the container: ml/ is NOT importable, so the server must find
    inference.py through the artifact."""
    assert "inference" not in sys.modules
    print("NOTE: serving env", json.dumps(model_environment()))

    from sagemaker_sklearn_container import serving
    from werkzeug.test import Client
    from werkzeug.wrappers import Response

    client = Client(serving.main, Response)

    def post(body, content_type="application/json"):
        r = client.post("/invocations", data=body, content_type=content_type)
        return r.status_code, json.loads(r.get_data(as_text=True))

    with open("/opt/ml/model/samples.json") as f:
        samples = json.load(f)
    for sample in samples:
        status, body = post(json.dumps(sample["reading"]))
        assert status == 200, (status, body)
        got = body["predictions"][0]["failure_probability"]
        # predict_proba sums trees across threads; allow last-digit rounding.
        assert abs(got - sample["expected_failure_probability"]) <= 1e-4, (got, sample)
        print("NOTE: %-13s -> %s" % (sample["name"], body["predictions"][0]))

    status, body = post(json.dumps({"instances": [s["reading"] for s in samples]}))
    assert status == 200 and len(body["predictions"]) == 3

    for bad, fragment in ((json.dumps({"temperature": 1, "vibration": 2}), "missing"),
                          ('{"temperature": NaN, "vibration": 2, "operating_hours": 3}', "NaN"),
                          ("not json", "not valid JSON")):
        status, body = post(bad)
        assert status == 400 and fragment in body["error"], (status, body)
        print("NOTE: 400 %s" % body["error"])
    status, body = post("1,2,3", "text/csv")
    assert status == 400, (status, body)

    imported = sys.modules["inference"].__file__
    print("NOTE: the container imported", imported)
    # The copy the server imported must be the one shipped in model.tar.gz,
    # never the checkout's ml/inference.py.
    with open(imported) as a, open("/opt/ml/model/code/inference.py") as b:
        assert a.read() == b.read()
    assert "/w/" not in imported, imported
    print("NOTE: serving stack smoke test passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        prepare()
        # The endpoint runs the server as a non-root user ("model-server"),
        # so pip cannot write the system site-packages and installs into
        # ~/.local instead. Reproduce that: serve as uid 1000 with its own
        # HOME, from /, with ml/ not importable.
        os.makedirs("/opt/ml/code", exist_ok=True)
        os.makedirs("/tmp/model-server", exist_ok=True)
        for path in ("/opt/ml", "/opt/ml/code", "/tmp/model-server"):
            os.chown(path, 1000, 1000)
        # The model's environment is in place before the process starts, as
        # it is for the container (PYTHONPATH only works that way).
        env = dict(os.environ, HOME="/tmp/model-server", **model_environment())
        subprocess.run([sys.executable, os.path.abspath(__file__), "--serve"], cwd="/",
                       user=1000, group=1000, env=env, check=True)
