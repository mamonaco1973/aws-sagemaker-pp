"""Shared fixtures: one real training run, reused by every test that needs it."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import dataset  # noqa: E402
import train  # noqa: E402


def write_channels(directory, flip_test_labels=False):
    """The same CSV layout SageMaker gives the container."""
    features, labels = dataset.generate()
    x_train, x_test, y_train, y_test = dataset.split(features, labels)
    if flip_test_labels:
        y_test = ["healthy" if y == "failure" else "failure" for y in y_test]
    for channel, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
        os.makedirs(os.path.join(directory, channel), exist_ok=True)
        with open(os.path.join(directory, channel, "%s.csv" % channel), "w") as f:
            f.write(dataset.to_csv(x, y))
    return x_train, x_test, y_train, y_test


def run_training(directory, flip_test_labels=False):
    write_channels(str(directory), flip_test_labels)
    model_dir = os.path.join(str(directory), "model")
    train.main(["--train", os.path.join(str(directory), "train"),
                "--test", os.path.join(str(directory), "test"),
                "--model_dir", model_dir])
    return model_dir


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """(model_dir, stdout of the training run) from a real local train.py run."""
    import contextlib
    import io

    directory = tmp_path_factory.mktemp("training")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        model_dir = run_training(directory)
    return model_dir, out.getvalue()
