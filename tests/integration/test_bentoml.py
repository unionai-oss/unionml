"""BentoML integration tests."""

import re
import runpy
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import docker
import pytest
import requests

BENTO_FILE = Path(__file__).parent / "bentoml_integration" / "bentofile.yaml"
HOST = "0.0.0.0"
URL = f"http://{HOST}"
DEFAULT_PORT = "3033"


def test_bentoml_build_containerize():
    # run the module to save a model to the bentoml model store

    # TODO: this should be a fixture
    app_vars = runpy.run_module("tests.integration.bentoml_integration.digits_classifier_app", run_name="__main__")
    saved_model = app_vars["saved_model"]

    # build the bento
    build_process = subprocess.Popen(["bentoml", "build", "-f", BENTO_FILE], stdout=subprocess.PIPE)
    out = build_process.stdout.read().decode()
    split_out = out.strip().split("\n")

    success_match = None
    for line in split_out:
        match = re.match(r"Successfully built Bento\(tag=\"(digits_classifier:[A-Za-z0-9]+)\"\)\.", line)
        if match:
            success_match = match

    assert success_match
    build_process.terminate()

    # containerize the bento with docker
    build_tag = success_match.group(1)
    subprocess.run(["bentoml", "containerize", "--load", build_tag])
    client = docker.from_env()

    bento_image = None
    for image in client.images.list():
        if build_tag in image.tags:
            bento_image = image
    assert bento_image is not None

    # cleanup
    subprocess.run(["bentoml", "models", "delete", "-y", str(saved_model.tag)])
    subprocess.run(["bentoml", "delete", "-y", build_tag])
    client.api.remove_image(build_tag)


def assert_health_check():
    exception = None
    for _ in range(30):
        try:
            health_check = requests.get(f"{URL}:{DEFAULT_PORT}/healthz")
            assert health_check.status_code == 200
            return
        except Exception as exc:  # pylint: disable=broad-except
            exception = exc
            time.sleep(1.0)
    raise RuntimeError(f"Health checks failed: {exception}")


@contextmanager
def _app(module: str):
    """Transient app server for testing."""
    process = subprocess.Popen(
        [
            "bentoml",
            "serve",
            f"tests.integration.bentoml_integration.{module}:service.svc",
            "--host",
            HOST,
            "--port",
            DEFAULT_PORT,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert_health_check()

    try:
        yield process
    finally:
        process.terminate()


@pytest.mark.parametrize(
    "module,is_async",
    [
        ["service", False],
        ["service_async", True],
    ],
)
def test_bentoml_serve(module, is_async):
    """Transient app server for testing."""

    app_vars = runpy.run_module("tests.integration.bentoml_integration.digits_classifier_app", run_name="__main__")
    saved_model = app_vars["saved_model"]

    with _app(module):
        api_request_vars = None
        try:
            for _ in range(30):
                try:
                    api_request_vars = runpy.run_module(
                        "tests.integration.bentoml_integration.api_requests", run_name="__main__"
                    )
                    break
                except Exception as exc:
                    print(f"Exception {exc}")
                    time.sleep(1.0)
            if api_request_vars is None:
                raise RuntimeError("Running the api request script failed.")
            prediction_response = api_request_vars["prediction_response"]
            output = prediction_response.json()
            assert len(output) == api_request_vars["n_samples"]
            assert all(isinstance(x, float) for x in output)
        finally:
            subprocess.run(["bentoml", "models", "delete", "-y", str(saved_model.tag)])
