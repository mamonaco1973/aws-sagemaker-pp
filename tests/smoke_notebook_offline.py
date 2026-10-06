"""Execute every notebook cell that does not call AWS, in the kernel's environment.

Not a pytest module. tests/run_local_checks.sh runs it in a Python 3.10 image
built from requirements.txt -- the notebook kernel's interpreter and pins --
against a model.tar.gz produced by the Python 3.12 training run. So it checks
two things a syntax check cannot:

  - the kernel can unpickle what the container pickled, and gets the same
    answers (samples.json holds the container-side probabilities)
  - the offline cells (setup, dataset, archive inspection, tree plot) run

Usage: python tests/smoke_notebook_offline.py /path/to/model.tar.gz
"""

import json
import os
import sys
import tarfile
import tempfile

import matplotlib

matplotlib.use("Agg")

import nbformat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
NOTEBOOK = os.path.join(ROOT, "notebook", "sagemaker_random_forest.ipynb")

# A cell containing any of these talks to AWS and is skipped.
AWS_MARKERS = ("get_caller_identity", "upload_dataset", "package_source", "start_training",
               "wait_for_training", "summary[", "list_objects_v2", "download_artifact",
               "workflow.deploy", "workflow.invoke", "endpoint_log_lines", "cleanup.cleanup")


def main(archive):
    work = tempfile.mkdtemp()
    config_path = os.path.join(work, "demo-config.json")
    with open(config_path, "w") as f:
        json.dump({"region": "us-east-1", "prefix": "offline", "bucket": "offline-bucket",
                   "execution_role_arn": "role", "sklearn_image_uri": "image",
                   "training_instance_type": "ml.m5.large",
                   "training_max_runtime_seconds": 900, "serverless_memory_mb": 2048,
                   "serverless_max_concurrency": 1}, f)
    os.environ["SAGEMAKER_DEMO_CONFIG"] = config_path

    model_dir = os.path.join(work, "model")
    with tarfile.open(archive) as tar:
        tar.extractall(model_dir, filter="data")

    # What the skipped AWS cells would have defined.
    namespace = {"job_name": "offline-train-1", "archive": archive, "model_dir": model_dir,
                 "__name__": "__main__"}
    # Keep the dataset cell's CSVs out of the checkout (this runs as root).
    from sagemaker_demo import workflow
    real_make_dataset = workflow.make_dataset
    workflow.make_dataset = lambda local_dir=None: real_make_dataset(os.path.join(work, "data"))
    os.chdir(os.path.join(ROOT, "notebook"))   # the notebook's own working directory
    ran = skipped = 0
    for i, cell in enumerate(nbformat.read(NOTEBOOK, as_version=4).cells):
        if cell.cell_type != "code":
            continue
        if any(marker in cell.source for marker in AWS_MARKERS):
            skipped += 1
            continue
        try:
            exec(compile(cell.source, "cell-%02d" % i, "exec"), namespace)
        except Exception:
            print("ERROR: cell %d failed:\n%s" % (i, cell.source))
            raise
        ran += 1
    print("NOTE: ran %d offline cells, skipped %d AWS cells" % (ran, skipped))

    # The kernel's copy of the model agrees with the container's answers.
    model, pd = namespace["model"], namespace["pd"]
    with open(os.path.join(model_dir, "samples.json")) as f:
        samples = json.load(f)
    col = list(model.classes_).index("failure")
    local = model.predict_proba(pd.DataFrame([s["reading"] for s in samples]))[:, col]
    for sample, p in zip(samples, local):
        assert abs(round(float(p), 4) - sample["expected_failure_probability"]) <= 1e-4, (p, sample)
    print("NOTE: Python %s kernel reproduces the Python 3.12 container's probabilities"
          % sys.version.split()[0])
    print("NOTE: notebook offline smoke test passed")


if __name__ == "__main__":
    main(sys.argv[1])
