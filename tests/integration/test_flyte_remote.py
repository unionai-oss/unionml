import importlib.util
import os
import runpy
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from flytekit.configuration import Config
from flytekit.remote import FlyteRemote

from unionml import Model

FLYTECTL_CMD = "sandbox" if os.getenv("RUN_CMDS_CI", False) else "demo"
NO_CLUSTER_MSG = "🛑 no demo cluster found" if FLYTECTL_CMD == "demo" else "🛑 no Sandbox found"


@contextmanager
def change_directory(path: Path):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


@pytest.fixture(scope="session")
def flyte_remote():
    p_status = subprocess.run(["flytectl", FLYTECTL_CMD, "status"], capture_output=True)

    cluster_preexists = True
    if p_status.stdout.decode().strip() == NO_CLUSTER_MSG:
        # if a demo cluster didn't exist already, then start one.
        cluster_preexists = False
        subprocess.run(["flytectl", FLYTECTL_CMD, "start"])

    remote = FlyteRemote(
        config=Config.auto(),
        default_project="flytesnacks",
        default_domain="development",
    )
    projects, *_ = remote.client.list_projects_paginated(limit=5, token=None)
    assert "flytesnacks" in [p.id for p in projects]
    assert "flytesnacks" in [p.name for p in projects]

    yield remote

    if not cluster_preexists:
        # only teardown the demo cluster if it didn't preexist
        subprocess.run(["flytectl", FLYTECTL_CMD, "teardown"])


def _import_model_from_file(module_name: str, file_path: Path) -> Model:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.model


@pytest.mark.parametrize(
    "ml_framework, filename",
    [
        ("sklearn", "model.joblib"),
        # ("pytorch", "model.pt"),
        # ("keras", "model.h5"),
    ],
    # ids=["sklearn", "pytorch", "keras"],
)
def test_unionml_deployment(flyte_remote: FlyteRemote, ml_framework, filename):
    model = _import_model_from_file(
        f"tests.integration.{ml_framework}_app.quickstart",
        Path(__file__).parent / f"{ml_framework}_app" / "quickstart.py",
    )
    with change_directory(Path(__file__).parent):
        model.remote(
            dockerfile="Dockerfile",
            project="unionml-sklearn-integration-test",
            domain="development",
        )
        app_version = str(uuid.uuid4())
        model.remote_deploy(app_version=app_version)
        model_artifact = model.remote_train(
            app_version=app_version,
            wait=True,
            hyperparameters={"C": 1.0, "max_iter": 10000},
        )
        import ipdb

        ipdb.set_trace()
        ...
