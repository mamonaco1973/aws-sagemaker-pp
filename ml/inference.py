"""Serving hooks for the SageMaker scikit-learn container.

Where things are inside the serverless endpoint's container:

    /opt/ml/model/                 model.tar.gz from the training job, extracted
    /opt/ml/model/model.joblib     the fitted RandomForestClassifier
    /opt/ml/model/metadata.json    feature order, classes, versions
    /opt/ml/model/code/inference.py  this file (SAGEMAKER_SUBMIT_DIRECTORY)

The container imports this module and calls the hooks in this order:

    model_fn(model_dir)                 once per serving process, at start-up
    input_fn(body, content_type)        once per request
    predict_fn(data, model)             once per request
    output_fn(prediction, accept)       once per request

No training happens here. The endpoint never sees the training data; it
only has the trees that training left behind in model.joblib.

Request body (application/json), one reading or up to 100:

    {"temperature": 81.5, "vibration": 4.2, "operating_hours": 9100}
    {"instances": [{...}, {...}]}

Response:

    {"predictions": [{"label": "failure", "failure_probability": 0.83}],
     "feature_order": ["temperature", "vibration", "operating_hours"]}

A malformed request gets HTTP 400 with {"error": "..."} naming the problem.
"""

import json
import logging
import math
import os
import time

import joblib
import pandas as pd
import sklearn

LOG = logging.getLogger(__name__)
JSON = "application/json"
MAX_INSTANCES = 100


class RequestError(ValueError):
    """A problem with the caller's request (HTTP 400), not with the model."""


def model_fn(model_dir):
    """Load the artifact. The container calls this once per worker process.

    This is the only joblib.load() in the serving path. Every request after
    it reuses the objects returned here.
    """
    started = time.time()
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # The fitted model remembers the column names it was trained on. The
    # metadata copy is what the request handler uses; they must agree.
    trained_on = list(getattr(model, "feature_names_in_", metadata["feature_order"]))
    if trained_on != metadata["feature_order"]:
        raise RuntimeError("metadata.json feature_order %s does not match the model's %s"
                           % (metadata["feature_order"], trained_on))

    trained_with = metadata.get("runtime", {}).get("scikit_learn")
    if trained_with and trained_with != sklearn.__version__:
        LOG.warning("model trained with scikit-learn %s, serving with %s",
                    trained_with, sklearn.__version__)

    LOG.info("joblib.load(%s) in pid %d took %.2fs (%d trees)", model_path, os.getpid(),
             time.time() - started, len(model.estimators_))
    return {"model": model, "metadata": metadata}


def parse_body(body):
    """Decode a JSON request into a list of readings (dicts not yet checked).

    Raises RequestError with a message meant for whoever sent the request.
    """
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except UnicodeDecodeError:
            raise RequestError("request body is not UTF-8 text")

    def reject_constant(name):
        # Python's json accepts NaN and Infinity, which are not JSON.
        raise RequestError("%s is not a valid number" % name)

    try:
        payload = json.loads(body, parse_constant=reject_constant)
    except RequestError:
        raise
    except ValueError as e:
        raise RequestError("request body is not valid JSON: %s" % e)

    if isinstance(payload, dict) and "instances" in payload:
        instances = payload["instances"]
        if not isinstance(instances, list) or not instances:
            raise RequestError('"instances" must be a non-empty list of readings')
    elif isinstance(payload, dict):
        instances = [payload]
    else:
        raise RequestError('send one reading as a JSON object, or {"instances": [...]}')
    if len(instances) > MAX_INSTANCES:
        raise RequestError("at most %d readings per request, got %d"
                           % (MAX_INSTANCES, len(instances)))
    return instances


def to_frame(instances, feature_order):
    """Check every reading and return a DataFrame in the training column order."""
    rows = []
    for i, instance in enumerate(instances):
        where = "reading %d" % i if len(instances) > 1 else "reading"
        if not isinstance(instance, dict):
            raise RequestError("%s must be a JSON object of named features" % where)
        missing = [name for name in feature_order if name not in instance]
        if missing:
            raise RequestError("%s is missing %s" % (where, ", ".join(missing)))
        # Unknown names are refused rather than ignored, so a typo such as
        # "temprature" fails loudly instead of predicting without that field.
        unknown = sorted(set(instance) - set(feature_order))
        if unknown:
            raise RequestError("%s has unknown fields %s; expected %s"
                               % (where, ", ".join(unknown), ", ".join(feature_order)))
        row = []
        # Iterating feature_order, not the request's own key order, is what
        # maps named fields onto the columns the trees were trained on.
        for name in feature_order:
            value = instance[name]
            # bool is a subclass of int in Python; true is not a temperature.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RequestError("%s: %s must be a number, got %s"
                                   % (where, name, json.dumps(value)))
            if not math.isfinite(value):
                raise RequestError("%s: %s must be finite" % (where, name))
            row.append(float(value))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(feature_order))


def input_fn(request_body, content_type):
    """Deserialize. Errors are returned, not raised.

    An exception here would reach the caller as a generic HTTP 500. Passing
    the error through to output_fn lets it answer 400 with the reason.
    """
    if (content_type or "").split(";")[0].strip().lower() != JSON:
        return RequestError("Content-Type must be %s, got %s" % (JSON, content_type))
    try:
        return parse_body(request_body)
    except RequestError as e:
        return e


def predict_fn(instances, loaded):
    """Validate against the model's own feature order, then predict."""
    if isinstance(instances, RequestError):
        return instances
    model = loaded["model"]
    metadata = loaded["metadata"]
    try:
        frame = to_frame(instances, metadata["feature_order"])
    except RequestError as e:
        return e
    failure_col = list(model.classes_).index(metadata["positive_label"])
    probabilities = model.predict_proba(frame)[:, failure_col]
    labels = model.predict(frame)
    return {
        "predictions": [
            {"label": str(label), "failure_probability": round(float(p), 4)}
            for label, p in zip(labels, probabilities)
        ],
        "feature_order": metadata["feature_order"],
    }


def output_fn(prediction, accept):
    if isinstance(prediction, RequestError):
        return _response({"error": str(prediction)}, 400)
    return _response(prediction, 200)


def _response(payload, status):
    body = json.dumps(payload)
    try:
        # Present in the SageMaker container; the only way to set a status.
        from sagemaker_containers.beta.framework import worker
    except ImportError:
        # Outside the container (local tests): the same body and status.
        return body, JSON, status
    return worker.Response(body, status=status, mimetype=JSON)
