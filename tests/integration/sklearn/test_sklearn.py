import runpy
import subprocess
import time
from contextlib import contextmanager

import pytest
import requests
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


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
    module_vars = runpy.run_module("tests.integration.sklearn.quickstart", run_name="__main__")
    trained_model = module_vars["trained_model"]
    predictions = module_vars["predictions"]

    assert isinstance(trained_model, LogisticRegression)
    check_is_fitted(trained_model)

    assert all([isinstance(x, float) and 0 <= x <= 9 for x in predictions])


def test_fastapi_app(tmp_path):
    # run the quickstart module to train a model
    model_path = tmp_path / "model.joblib"
    module_vars = runpy.run_module("tests.integration.sklearn.quickstart", run_name="__main__")

    # extract fklearn model and trained_model from module global namespace
    model = module_vars["model"]
    trained_model = module_vars["trained_model"]

    model.save(trained_model, model_path)
    n_samples = 5

    with contextmanager(_app)("--model-path", str(model_path)):
        api_request_vars = runpy.run_module("tests.integration.sklearn.api_requests", run_name="__main__")
        prediction_response = api_request_vars["prediction_response"]
        output = prediction_response.json()
        assert len(output) == n_samples
        assert all(isinstance(x, float) for x in output)


def test_fastapi_app_no_model():
    digits = load_digits(as_frame=True)
    features = digits.frame[digits.feature_names]
    n_samples = 5

    # excluding the --model-path argument should raise an error since the fklearn.Model object
    # doesn't have a model_artifact attribute set yet
    with contextmanager(_app)(port="8001"):
        prediction_response = requests.post(
            "http://127.0.0.1:8001/predict?local=True",
            json={"features": features.sample(n_samples, random_state=42).to_dict(orient="records")},
        )
        assert prediction_response.json() == {"detail": "trained model not found"}
