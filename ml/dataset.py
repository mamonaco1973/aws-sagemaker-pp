"""Synthetic equipment sensor data for the random forest demo.

This is demonstration data, not a validated equipment failure model. The
labels come from a made-up probability formula plus noise, so the classifier
has something real to learn but can never be perfect: two machines with
identical readings can get different labels. That is deliberate. A dataset a
model can score 100% on teaches nothing about evaluation.

Only numpy and the standard library are used, so the same file runs in the
SageMaker container, in the notebook, and in the CLI.
"""

import csv
import io

import numpy as np

FEATURES = ("temperature", "vibration", "operating_hours")
TARGET = "status"
LABELS = ("healthy", "failure")
POSITIVE_LABEL = "failure"

# One seed for the training data and a different one for "new" readings, so
# the readings sent to the endpoint were never seen during training.
TRAINING_SEED = 42
NEW_READINGS_SEED = 2026


def failure_probability(temperature, vibration, operating_hours):
    """The hidden rule the model is meant to rediscover.

    Hot, shaky, old machines fail more often; any one of the three alone
    raises the odds a little, and heat plus vibration together raises them
    more. The constant keeps failures a minority (a little over one in four).
    """
    t = (np.asarray(temperature, dtype=float) - 70.0) / 8.0
    v = (np.asarray(vibration, dtype=float) - 3.0) / 1.5
    h = (np.asarray(operating_hours, dtype=float) - 6000.0) / 3500.0
    logit = -1.8 + 2.0 * t + 2.0 * v + 1.2 * h + 0.5 * t * v
    return 1.0 / (1.0 + np.exp(-logit))


def generate(n_rows=5000, seed=TRAINING_SEED):
    """Return (features, labels): an (n, 3) float array and a list of strings.

    Deterministic for a given seed and numpy version (numpy is pinned in
    every requirements file for this reason).
    """
    rng = np.random.default_rng(seed)
    temperature = rng.normal(70.0, 8.0, n_rows)
    vibration = rng.lognormal(np.log(2.8), 0.45, n_rows)
    operating_hours = rng.uniform(0.0, 12000.0, n_rows)

    # Sensors are not perfect: the label is drawn from the true readings, but
    # the model only ever sees noisy measurements of them.
    p = failure_probability(temperature, vibration, operating_hours)
    is_failure = rng.random(n_rows) < p

    measured = np.column_stack([
        temperature + rng.normal(0.0, 1.5, n_rows),
        np.clip(vibration + rng.normal(0.0, 0.3, n_rows), 0.05, None),
        operating_hours,
    ])
    measured = np.round(measured, 2)
    labels = [LABELS[1] if f else LABELS[0] for f in is_failure]
    return measured, labels


def split(features, labels, test_size=0.2, seed=TRAINING_SEED):
    """Stratified train/test split: both halves keep the same failure rate.

    The split happens before upload, and the halves travel as separate
    SageMaker channels, so the held-out rows physically never reach the code
    that calls fit().
    """
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels,
        random_state=seed)
    return x_train, x_test, list(y_train), list(y_test)


def to_csv(features, labels=None):
    """CSV text with a header row; the label column is omitted when None."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    header = list(FEATURES) + ([TARGET] if labels is not None else [])
    writer.writerow(header)
    for i, row in enumerate(features):
        values = ["%.2f" % value for value in row]
        if labels is not None:
            values.append(labels[i])
        writer.writerow(values)
    return buffer.getvalue()


def as_records(features):
    """Rows as JSON-ready dicts keyed by feature name."""
    return [
        {name: float(value) for name, value in zip(FEATURES, row)}
        for row in features
    ]
