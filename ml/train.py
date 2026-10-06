"""Training entry point, run by the SageMaker scikit-learn container.

SageMaker starts this as `python train.py --n_estimators 200 ...` (one flag
per hyperparameter) after copying the S3 channels into the container:

    /opt/ml/input/data/train/train.csv   <- s3://<bucket>/data/train/
    /opt/ml/input/data/test/test.csv     <- s3://<bucket>/data/test/

Everything written to /opt/ml/model is packed into model.tar.gz and uploaded
to S3 when the script exits. Then the training instance is shut down: the
model lives on only as that archive.

The same script runs locally (tests, or `python ml/train.py --help`), because
every path comes from an argument whose default is SageMaker's env variable.
"""

import argparse
import json
import os
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)

import dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    # Hyperparameters. Fixed in advance, not tuned: there is no validation
    # split here, and tuning against the test set would make its score a lie.
    parser.add_argument("--n_estimators", type=int, default=200)
    # Bounded trees keep the artifact small (about a megabyte compressed)
    # and stop each tree from memorising individual noisy rows.
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    return parser.parse_args(argv)


def read_channel(directory):
    """Every CSV in a channel directory, in the training feature order."""
    files = sorted(f for f in os.listdir(directory) if f.endswith(".csv"))
    if not files:
        raise SystemExit("ERROR: no CSV files in %s" % directory)
    frame = pd.concat([pd.read_csv(os.path.join(directory, f)) for f in files])
    missing = [c for c in dataset.FEATURES + (dataset.TARGET,) if c not in frame]
    if missing:
        raise SystemExit("ERROR: %s is missing columns %s" % (directory, missing))
    return frame[list(dataset.FEATURES)], frame[dataset.TARGET]


def evaluate(model, x_test, y_test):
    predicted = model.predict(x_test)
    # Rows are actual, columns predicted, in this fixed order.
    order = list(dataset.LABELS)
    return {
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, pos_label=dataset.POSITIVE_LABEL)),
        "recall": float(recall_score(y_test, predicted, pos_label=dataset.POSITIVE_LABEL)),
        "confusion_matrix": {
            "labels": order,
            "matrix": confusion_matrix(y_test, predicted, labels=order).tolist(),
        },
    }


def failure_column(model):
    """Index of 'failure' in predict_proba's columns (classes_ is sorted)."""
    return list(model.classes_).index(dataset.POSITIVE_LABEL)


def sample_readings(model):
    """Three new readings that span the model's range of answers.

    Scores 200 readings the model has never seen and keeps the least risky,
    the one closest to a coin flip, and the most risky. The probabilities
    stored here are what this model actually said, so the endpoint's answers
    can be checked against them rather than against a claim in the docs.
    """
    features, _ = dataset.generate(n_rows=200, seed=dataset.NEW_READINGS_SEED)
    frame = pd.DataFrame(features, columns=list(dataset.FEATURES))
    p = model.predict_proba(frame)[:, failure_column(model)]
    picks = [("lowest risk", int(np.argmin(p))),
             ("borderline", int(np.argmin(np.abs(p - 0.5)))),
             ("highest risk", int(np.argmax(p)))]
    return [{"name": name,
             "reading": dataset.as_records(features[i:i + 1])[0],
             "expected_failure_probability": round(float(p[i]), 4)}
            for name, i in picks]


def main(argv=None):
    args = parse_args(argv)
    started = time.time()

    x_train, y_train = read_channel(args.train)
    x_test, y_test = read_channel(args.test)
    print("NOTE: %d training rows, %d held-out test rows" % (len(x_train), len(x_test)))
    print("NOTE: training failure rate %.3f" % (y_train == dataset.POSITIVE_LABEL).mean())

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )
    # A DataFrame, so the fitted model records feature_names_in_ and will
    # refuse a later DataFrame whose columns are in a different order.
    model.fit(x_train, y_train)

    # The first and only time the model meets the test rows.
    metrics = evaluate(model, x_test, y_test)
    # One line per metric, matched by the MetricDefinitions regexes in
    # sagemaker_demo/workflow.py, so DescribeTrainingJob reports them too.
    for name in ("accuracy", "precision", "recall"):
        print("METRIC %s=%.4f" % (name, metrics[name]))
    # The four confusion-matrix cells too, so the whole evaluation is on the
    # training job's record without opening the artifact.
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]["matrix"]
    for name, count in (("true_healthy", tn), ("false_failure", fp),
                        ("missed_failure", fn), ("true_failure", tp)):
        print("METRIC %s=%d" % (name, count))
    print("NOTE: confusion matrix (rows actual, cols predicted, %s): %s"
          % ("/".join(dataset.LABELS), metrics["confusion_matrix"]["matrix"]))

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    # compress=3: the trees are highly repetitive arrays, and a smaller
    # archive means a faster serverless cold start.
    joblib.dump(model, model_path, compress=3)

    metadata = {
        "feature_order": list(dataset.FEATURES),
        "classes": [str(c) for c in model.classes_],
        "positive_label": dataset.POSITIVE_LABEL,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
        },
        "training_rows": int(len(x_train)),
        "feature_importances": {
            name: round(float(v), 4)
            for name, v in zip(dataset.FEATURES, model.feature_importances_)
        },
        # The artifact is a pickle of scikit-learn objects, so it can only be
        # loaded by a compatible scikit-learn. Recording the versions makes a
        # mismatch diagnosable instead of mysterious.
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "training_job": os.environ.get("TRAINING_JOB_NAME", "local"),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(args.model_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.model_dir, "samples.json"), "w") as f:
        json.dump(sample_readings(model), f, indent=2)

    # The serving code travels inside the artifact (model.tar.gz:code/), so
    # the archive alone is everything an endpoint needs besides the image.
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(os.path.join(HERE, "inference.py"), code_dir)

    size_kb = os.path.getsize(model_path) / 1024.0
    print("NOTE: wrote %s (%.0f KB) in %.1fs" % (model_path, size_kb, time.time() - started))


if __name__ == "__main__":
    main()
