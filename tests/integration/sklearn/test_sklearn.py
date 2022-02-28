import re
import subprocess
import sys
import time
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
