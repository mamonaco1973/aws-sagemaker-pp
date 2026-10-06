"""The serving hooks, called in the order the container calls them."""

import json
import os
import shutil

import pandas as pd
import pytest

import inference

GOOD = {"temperature": 81.5, "vibration": 4.2, "operating_hours": 9100}


@pytest.fixture(scope="module")
def loaded(trained):
    model_dir, _ = trained
    return inference.model_fn(model_dir)


def handle(loaded, body, content_type="application/json"):
    """input_fn -> predict_fn -> output_fn, as the container runs them."""
    data = inference.input_fn(body, content_type)
    prediction = inference.predict_fn(data, loaded)
    response = inference.output_fn(prediction, "application/json")
    if isinstance(response, tuple):      # no sagemaker_containers installed
        response_body, mimetype, status = response
    else:                                # the container's Flask response
        response_body, mimetype, status = (response.get_data(as_text=True),
                                           response.mimetype, response.status_code)
    assert mimetype == "application/json"
    return status, json.loads(response_body)


def direct_probability(loaded, reading):
    model = loaded["model"]
    frame = pd.DataFrame([reading])[list(model.feature_names_in_)]
    return float(model.predict_proba(frame)[0, list(model.classes_).index("failure")])


def test_one_reading(loaded):
    status, body = handle(loaded, json.dumps(GOOD))
    assert status == 200
    (result,) = body["predictions"]
    assert result["label"] in ("healthy", "failure")
    assert result["failure_probability"] == round(direct_probability(loaded, GOOD), 4)
    assert body["feature_order"] == ["temperature", "vibration", "operating_hours"]


def test_bytes_body_and_charset_are_accepted(loaded):
    status, _ = handle(loaded, json.dumps(GOOD).encode(), "application/json; charset=utf-8")
    assert status == 200


def test_fields_are_mapped_by_name_not_position(loaded):
    reordered = {"operating_hours": 9100, "vibration": 4.2, "temperature": 81.5}
    _, a = handle(loaded, json.dumps(GOOD))
    _, b = handle(loaded, json.dumps(reordered))
    assert a == b
    # Feeding the same numbers positionally in the wrong columns would give a
    # different answer -- which is why the mapping matters.
    swapped = {"temperature": 4.2, "vibration": 81.5, "operating_hours": 9100}
    _, c = handle(loaded, json.dumps(swapped))
    assert c["predictions"] != a["predictions"]


def test_batch_and_integer_values(loaded):
    low = {"temperature": 60, "vibration": 2, "operating_hours": 500}
    status, body = handle(loaded, json.dumps({"instances": [low, GOOD]}))
    assert status == 200
    probs = [p["failure_probability"] for p in body["predictions"]]
    assert probs == [round(direct_probability(loaded, r), 4) for r in (low, GOOD)]
    assert probs[0] < probs[1]


def test_label_agrees_with_probability(loaded):
    for reading in ({"temperature": 55, "vibration": 1.5, "operating_hours": 100},
                    {"temperature": 90, "vibration": 6.0, "operating_hours": 11500}):
        _, body = handle(loaded, json.dumps(reading))
        p = body["predictions"][0]
        assert p["label"] == ("failure" if p["failure_probability"] > 0.5 else "healthy")


@pytest.mark.parametrize("body, fragment", [
    (json.dumps({"temperature": 70, "vibration": 3}), "missing operating_hours"),
    (json.dumps(dict(GOOD, temprature=70)), "unknown fields temprature"),
    (json.dumps(dict(GOOD, temperature="hot")), 'temperature must be a number, got "hot"'),
    (json.dumps(dict(GOOD, vibration=True)), "vibration must be a number, got true"),
    (json.dumps(dict(GOOD, vibration=None)), "vibration must be a number, got null"),
    (json.dumps(dict(GOOD, temperature=float("nan"))), "NaN is not a valid number"),
    (json.dumps(dict(GOOD, temperature=float("inf"))), "Infinity is not a valid number"),
    (json.dumps(dict(GOOD, operating_hours=1e400)), "Infinity is not a valid number"),
    ('{"temperature": 70,', "not valid JSON"),
    (json.dumps([GOOD]), "send one reading as a JSON object"),
    (json.dumps({"instances": []}), "non-empty list"),
    (json.dumps({"instances": [GOOD, 5]}), "reading 1 must be a JSON object"),
    (json.dumps({"instances": [GOOD] * 101}), "at most 100 readings"),
    (b"\xff\xfe", "not UTF-8"),
], ids=["missing", "unknown", "string", "bool", "null", "nan", "inf", "overflow",
        "bad-json", "list", "empty", "non-object", "too-many", "not-utf8"])
def test_bad_requests_get_400_with_a_reason(loaded, body, fragment):
    status, response = handle(loaded, body)
    assert status == 400
    assert fragment in response["error"]


def test_wrong_content_type_is_400(loaded):
    status, response = handle(loaded, "70,3,9100", "text/csv")
    assert status == 400 and "Content-Type must be application/json" in response["error"]


def test_huge_number_overflowing_to_inf_is_rejected(loaded):
    # 1e400 is valid JSON syntax but not a finite float.
    status, _ = handle(loaded, '{"temperature": 1e400, "vibration": 3, "operating_hours": 1}')
    assert status == 400


def test_model_fn_refuses_metadata_that_disagrees_with_the_model(trained, tmp_path):
    model_dir, _ = trained
    broken = tmp_path / "model"
    shutil.copytree(model_dir, broken)
    with open(broken / "metadata.json") as f:
        metadata = json.load(f)
    metadata["feature_order"] = ["vibration", "temperature", "operating_hours"]
    with open(broken / "metadata.json", "w") as f:
        json.dump(metadata, f)
    with pytest.raises(RuntimeError, match="does not match"):
        inference.model_fn(str(broken))


def test_inference_imports_nothing_from_training():
    """The endpoint gets only code/inference.py from the archive."""
    source = open(os.path.join(os.path.dirname(inference.__file__), "inference.py")).read()
    assert "import dataset" not in source and "import train" not in source
