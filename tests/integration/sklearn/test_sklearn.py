import re
import runpy
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
from sklearn.datasets import load_digits


def _app(*args, port: str = "8000"):
    """Transient app server for testing."""
    process = subprocess.Popen(
        ["fklearn", "serve", "tests.integration.sklearn.fastapi_app:app", "--port", port, *args],
        stdout=subprocess.PIPE,
    )
    _wait_to_exist(port)
    try:
        yield process
    finally:
        process.terminate()


@pytest.fixture(scope="function")
def app():
    yield from _app()


def _wait_to_exist(port):
    for _ in range(30):
        try:
            requests.get(f"http://127.0.0.1:{port}/")
            break
        except Exception:  # pylint: disable=broad-except
            time.sleep(3.0)


def test_module(capfd):
    module_path = Path(__file__)
    subprocess.run(
        [
            sys.executable,
            str(module_path.parent / "quickstart.py"),
        ],
        text=True,
    )
    cap = capfd.readouterr()
    expected_patterns = [
        r"LogisticRegression\(max_iter=1000.0\)",
        r"\{'train': 1.0, 'test': [0-9.]+\}",
        r"\[6\.0, 9\.0, 3\.0, 7\.0, 2\.0\]",
    ]

    assert cap.out.strip() != ""

    for patt, line in zip(expected_patterns, cap.out.strip().split("\n")):
        assert re.match(patt, line)


def test_fastapi_app(app, capfd):
    module_path = Path(__file__)
    subprocess.run(
        [
            sys.executable,
            str(module_path.parent / "api_requests.py"),
        ],
        text=True,
    )
    cap = capfd.readouterr()
    expected_patterns = [
        r'\{"trained_model":"LogisticRegression\(max_iter=1000.0\)","metrics":\{"train":1.0,"test":[0-9.]+\},"flyte_execution_id":null\}',  # noqa
        r"\[6\.0,9\.0,3\.0,7\.0,2\.0\]",
    ]
    for patt, line in zip(expected_patterns, cap.out.strip().split("\n")):
        assert re.match(patt, line)


def test_load_model_from_local_fs(tmp_path):
    digits = load_digits(as_frame=True)
    features = digits.frame[digits.feature_names]

    # run the quickstart module to train a model
    model_path = tmp_path / "model.joblib"
    module_vars = runpy.run_module("tests.integration.sklearn.quickstart", run_name="__main__")

    # extract fklearn model and trained_model from module global namespace
    model = module_vars["model"]
    trained_model = module_vars["trained_model"]

    model.save(trained_model, model_path)
    n_samples = 5

    with contextmanager(_app)("--model-path", str(model_path), port="8001"):
        prediction_response = requests.post(
            "http://127.0.0.1:8001/predict?local=True",
            json={"features": features.sample(n_samples, random_state=42).to_dict(orient="records")},
        )
        output = prediction_response.json()
        assert len(output) == n_samples
        assert all(isinstance(x, float) for x in output)

    # excluding the --model-path argument should raise an error since the fklearn.Model object
    # doesn't have a _latest_model attribute set yet
    with contextmanager(_app)(port="8002"):
        prediction_response = requests.post(
            "http://127.0.0.1:8002/predict?local=True",
            json={"features": features.sample(n_samples, random_state=42).to_dict(orient="records")},
        )
        assert prediction_response.json() == {"detail": "trained model not found"}
