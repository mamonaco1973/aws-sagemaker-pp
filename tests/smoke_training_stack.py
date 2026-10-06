"""Run train.py through the scikit-learn container's own training entry point.

Not a pytest module: it needs the image tests/run_local_checks.sh builds,
with sagemaker-training 4.8.0 and sagemaker-sklearn-container v1.4-2-py312.

It writes /opt/ml/input exactly as SageMaker does for the CreateTrainingJob
request workflow.training_job_request() builds -- the same hyperparameters,
JSON-encoded the same way, the same two channels -- with one substitution:
sagemaker_submit_directory names a local copy of the source tarball instead
of its S3 URI. Then it calls the container's training main() and checks
/opt/ml/model.
"""

import json
import os
import shutil
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def main():
    from conftest import write_channels
    from sagemaker_demo import workflow
    from sagemaker_demo.config import Config

    for d in ("/opt/ml/input", "/opt/ml/model", "/opt/ml/output", "/opt/ml/code"):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs("/opt/ml/input/config")
    os.makedirs("/opt/ml/model")
    os.makedirs("/opt/ml/output/data")
    write_channels("/opt/ml/input/data")

    source = "/tmp/sourcedir.tar.gz"
    with open(source, "wb") as f:
        f.write(workflow.source_tarball())

    cfg = Config(region="us-east-1", prefix="smoke", bucket="bucket",
                 execution_role_arn="x", sklearn_image_uri="x",
                 training_instance_type="ml.m5.large", training_max_runtime_seconds=900,
                 serverless_memory_mb=2048, serverless_max_concurrency=1)
    request = workflow.training_job_request(
        cfg, "smoke-train-1", "s3://bucket/code/sourcedir.tar.gz",
        {"train": "s3://bucket/data/train/", "test": "s3://bucket/data/test/"})

    hyperparameters = dict(request["HyperParameters"])
    hyperparameters["sagemaker_submit_directory"] = json.dumps(source)
    with open("/opt/ml/input/config/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f)
    with open("/opt/ml/input/config/inputdataconfig.json", "w") as f:
        json.dump({c["ChannelName"]: {"ContentType": c["ContentType"],
                                      "TrainingInputMode": "File",
                                      "S3DistributionType": "FullyReplicated",
                                      "RecordWrapperType": "None"}
                   for c in request["InputDataConfig"]}, f)
    with open("/opt/ml/input/config/resourceconfig.json", "w") as f:
        json.dump({"current_host": "algo-1", "hosts": ["algo-1"],
                   "network_interface_name": "eth0"}, f)
    os.environ["TRAINING_JOB_NAME"] = request["TrainingJobName"]
    print("NOTE: hyperparameters.json", json.dumps(hyperparameters))

    from sagemaker_sklearn_container.training import main as container_main
    container_main()

    files = sorted(os.path.relpath(os.path.join(d, f), "/opt/ml/model")
                   for d, _, names in os.walk("/opt/ml/model") for f in names)
    print("NOTE: /opt/ml/model ->", files)
    for expected in ("model.joblib", "metadata.json", "evaluation.json", "samples.json",
                     "code/inference.py"):
        assert expected in files, expected
    with open("/opt/ml/model/metadata.json") as f:
        metadata = json.load(f)
    assert metadata["hyperparameters"]["n_estimators"] == 200, metadata["hyperparameters"]
    assert metadata["training_job"] == "smoke-train-1"
    # What SageMaker would pack: everything under /opt/ml/model.
    with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
        tar.add("/opt/ml/model", arcname=".")
    print("NOTE: model.tar.gz would be %d KB" % (os.path.getsize("/tmp/model.tar.gz") // 1024))
    print("NOTE: training stack smoke test passed")


if __name__ == "__main__":
    main()
