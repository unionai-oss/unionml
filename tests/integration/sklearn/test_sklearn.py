import subprocess
import sys
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="module")
def app():
    """Transient app server for testing."""
    process = subprocess.Popen(
        ["uvicorn", "tests.integration.sklearn.fastapi_app:app", "--port", "8000"],
        stdout=subprocess.PIPE,
    )
    _wait_to_exist()
    yield process
    process.terminate()


def _wait_to_exist():
    for _ in range(20):
        try:
            requests.post("http://127.0.0.1:8000/")
            break
        except Exception:  # pylint: disable=broad-except
            time.sleep(3.0)


def test_module():
    module_path = Path(__file__)
    output = subprocess.run(
        [
            sys.executable,
            str(module_path.parent / "quickstart.py"),
        ],
        capture_output=True,
    )
    expected_output = (
        "LogisticRegression(max_iter=1000.0) {'train': 1.0, 'test': 0.9722222222222222} [6.0, 9.0, 3.0, 7.0, 2.0]"
    )
    assert output.stdout.decode().strip() == expected_output


def test_fastapi_app(app):
    module_path = Path(__file__)
    output = subprocess.run(
        [
            sys.executable,
            str(module_path.parent / "api_requests.py"),
        ],
        capture_output=True,
    )
    output_lines = output.stdout.decode().strip().split("\n")
    expected = [
        '{"trained_model":"LogisticRegression(max_iter=1000.0)","metrics":{"train":1.0,"test":0.9722222222222222},"flyte_execution_id":null}',  # noqa
        "[6.0,9.0,3.0,7.0,2.0]",
    ]
    assert output_lines == expected
