"""BentoML integration tests."""

import re
import runpy
import subprocess
import time
from pathlib import Path

import docker
import requests

BENTO_FILE = Path(__file__).parent / "bentoml_integration" / "bentofile.yaml"
URL = "http://127.0.0.1"
DEFAULT_PORT = "3000"


def test_bentoml_build_containerize():
    # build the bento
    build_process = subprocess.Popen(["bentoml", "build", "-f", BENTO_FILE], stdout=subprocess.PIPE)
    out = build_process.stdout.read().decode()
    split_out = out.strip().split("\n")

    success_match = re.match(
        r"Successfully built Bento\(tag=\"(unionml_digits_classifier:[A-Za-z0-9]+)\"\)\.",
        split_out[-1],
    )
    assert success_match
    build_process.terminate()

    # containerize the bento with docker
    build_tag = success_match.group(1)
    subprocess.run(["bentoml", "containerize", build_tag])
    client = docker.from_env()

    bento_image = None
    for image in client.images.list():
        if build_tag in image.tags:
            bento_image = image
    assert bento_image is not None

    # cleanup
    subprocess.run(["bentoml", "delete", "-y", build_tag])
    client.api.remove_image(build_tag)


def assert_health_check():
    for _ in range(30):
        try:
            health_check = requests.get(f"{URL}:{DEFAULT_PORT}/healthz")
            assert health_check.json()["status"] == 200
        except Exception:  # pylint: disable=broad-except
            time.sleep(1.0)


def test_bentoml_serve():
    """Transient app server for testing."""
    process = subprocess.Popen(
        [
            "bentoml",
            "serve",
            "tests.integration.bentoml_integration.service:bentoml_service.svc",
            "--port",
            DEFAULT_PORT,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert_health_check()

    api_request_vars = None
    try:
        for _ in range(10):
            # for some reason the keras test has trouble connecting to the fastapi app
            try:
                api_request_vars = runpy.run_module(
                    "tests.integration.bentoml_integration.api_requests", run_name="__main__"
                )
                break
            except Exception:
                time.sleep(1.0)
        if api_request_vars is None:
            raise RuntimeError("Running the api request script failed.")
        prediction_response = api_request_vars["prediction_response"]
        output = prediction_response.json()
        assert len(output) == api_request_vars["n_samples"]
        assert all(isinstance(x, float) for x in output)
    finally:
        process.terminate()
