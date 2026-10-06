"""Notebook structure: valid, in sync with its generator, and calling real APIs."""

import ast
import json
import os
import re
import subprocess
import sys

import nbformat

from conftest import ROOT

NOTEBOOK = os.path.join(ROOT, "notebook", "sagemaker_random_forest.ipynb")

SECTIONS = [
    "Architecture and terminology", "Generate and inspect the dataset",
    "Upload the data", "Launch the training job", "Evaluation results",
    "Locate and download the model artifact", "Inspect the archive",
    "Look inside one tree", "Deploy to Serverless Inference",
    "Send new readings", "Clean up",
]


def load():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_valid_nbformat_with_the_pinned_kernel():
    nb = load()
    nbformat.validate(nb)
    assert nb.metadata.kernelspec.name == "sagemaker-demo"


def test_eleven_sections_in_order():
    headings = [line[3:] for c in load().cells if c.cell_type == "markdown"
                for line in c.source.splitlines() if line.startswith("## ")]
    numbered = [h.split(". ", 1) for h in headings if re.match(r"\d+\. ", h)]
    assert [int(n) for n, _ in numbered] == list(range(1, 12))
    for (_, title), expected in zip(numbered, SECTIONS):
        assert title.startswith(expected), (title, expected)


def test_every_code_cell_parses_and_has_no_outputs():
    for cell in load().cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None


def test_no_pasted_credentials_or_arns():
    text = "\n".join(c.source for c in load().cells)
    assert not re.search(r"arn:aws:", text)
    assert not re.search(r"AKIA[0-9A-Z]{16}", text)
    assert "aws_secret_access_key" not in text.lower()


def test_cells_only_call_functions_that_exist():
    from sagemaker_demo import cleanup, config, workflow
    import inference
    modules = {"workflow": workflow, "cleanup": cleanup, "config": config, "inference": inference}
    for cell in load().cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in modules:
                assert hasattr(modules[node.value.id], node.attr), \
                    "%s.%s" % (node.value.id, node.attr)


def test_notebook_matches_its_generator(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, os.path.join(ROOT, "make_notebook.py"), str(out)],
                   check=True, capture_output=True)
    with open(NOTEBOOK) as a, open(out) as b:
        assert json.load(a) == json.load(b), "run python3 make_notebook.py"
