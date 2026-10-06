import json
import os
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import dataset
from conftest import ROOT, run_training


def test_artifact_layout(trained):
    model_dir, _ = trained
    for name in ("model.joblib", "metadata.json", "evaluation.json", "samples.json",
                 "code/inference.py"):
        assert os.path.isfile(os.path.join(model_dir, name)), name


def test_bounded_forest_and_small_artifact(trained):
    model_dir, _ = trained
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    assert isinstance(model, RandomForestClassifier)
    assert len(model.estimators_) == 200
    assert max(t.get_depth() for t in model.estimators_) <= 8
    assert os.path.getsize(os.path.join(model_dir, "model.joblib")) < 3 * 1024 * 1024


def test_model_learned_something_but_is_not_perfect(trained):
    model_dir, _ = trained
    with open(os.path.join(model_dir, "evaluation.json")) as f:
        evaluation = json.load(f)
    (tn, fp), (fn, tp) = evaluation["confusion_matrix"]["matrix"]
    majority_baseline = (tn + fp) / evaluation["test_rows"]   # always "healthy"
    assert evaluation["accuracy"] > majority_baseline + 0.05
    assert evaluation["accuracy"] < 0.97
    assert tn + fp + fn + tp == evaluation["test_rows"] == 1000
    assert abs(evaluation["recall"] - tp / (tp + fn)) < 1e-9
    assert abs(evaluation["precision"] - tp / (tp + fp)) < 1e-9


def test_test_rows_cannot_influence_the_fit(tmp_path):
    """Flip every held-out label: the fitted model must not change at all."""
    normal = joblib.load(os.path.join(run_training(tmp_path / "a"), "model.joblib"))
    flipped = joblib.load(os.path.join(run_training(tmp_path / "b", flip_test_labels=True),
                                       "model.joblib"))
    # Compare the fitted trees themselves: every split feature, threshold and
    # leaf count. (predict_proba sums trees across threads, so its last bits
    # vary between calls even for one model.)
    for a, b in zip(normal.estimators_, flipped.estimators_):
        assert np.array_equal(a.tree_.feature, b.tree_.feature)
        assert np.array_equal(a.tree_.threshold, b.tree_.threshold)
        assert np.array_equal(a.tree_.value, b.tree_.value)
    probe, _ = dataset.generate(n_rows=300, seed=7)
    probe = pd.DataFrame(probe, columns=list(dataset.FEATURES))
    np.testing.assert_allclose(normal.predict_proba(probe), flipped.predict_proba(probe),
                               rtol=0, atol=1e-12)


def test_metric_lines_match_the_training_job_regexes(trained):
    """SageMaker scrapes the log with workflow.METRIC_DEFINITIONS."""
    _, stdout = trained
    from sagemaker_demo.workflow import METRIC_DEFINITIONS

    model_dir, _ = trained
    with open(os.path.join(model_dir, "evaluation.json")) as f:
        evaluation = json.load(f)
    scraped = {}
    for definition in METRIC_DEFINITIONS:
        match = re.search(definition["Regex"], stdout)
        assert match, "no log line for %s" % definition["Name"]
        scraped[definition["Name"]] = float(match.group(1))
    assert abs(scraped["test:accuracy"] - evaluation["accuracy"]) < 1e-4
    (tn, fp), (fn, tp) = evaluation["confusion_matrix"]["matrix"]
    assert (scraped["test:true_healthy"], scraped["test:false_failure"],
            scraped["test:missed_failure"], scraped["test:true_failure"]) == (tn, fp, fn, tp)


def test_samples_span_the_range_and_match_a_fresh_reload(trained):
    """Reload in a separate interpreter: nothing from training is in memory."""
    model_dir, _ = trained
    with open(os.path.join(model_dir, "samples.json")) as f:
        samples = json.load(f)
    p = [s["expected_failure_probability"] for s in samples]
    assert [s["name"] for s in samples] == ["lowest risk", "borderline", "highest risk"]
    assert p[0] < 0.3 < p[1] < 0.7 < p[2]

    script = (
        "import json, sys, joblib, pandas as pd\n"
        "m = joblib.load(sys.argv[1] + '/model.joblib')\n"
        "s = json.load(open(sys.argv[1] + '/samples.json'))\n"
        "col = list(m.classes_).index('failure')\n"
        "print(json.dumps([round(float(x), 4) for x in "
        "m.predict_proba(pd.DataFrame([x['reading'] for x in s]))[:, col]]))\n")
    out = subprocess.run([sys.executable, "-c", script, model_dir],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == p


def test_metadata_records_feature_order_and_versions(trained):
    model_dir, _ = trained
    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    assert metadata["feature_order"] == list(model.feature_names_in_) == list(dataset.FEATURES)
    assert metadata["classes"] == list(model.classes_)
    import sklearn
    assert metadata["runtime"]["scikit_learn"] == sklearn.__version__


def test_pinned_versions_agree_across_requirement_files():
    """The notebook kernel must be able to unpickle what the container wrote."""
    def pins(path):
        with open(os.path.join(ROOT, path)) as f:
            return dict(line.strip().split("==") for line in f
                        if "==" in line and not line.startswith("#"))
    container, kernel = pins("ml/requirements.txt"), pins("requirements.txt")
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "scipy"):
        assert container[package] == kernel[package], package
