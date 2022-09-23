import importlib.util
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import pytest
from flytekit.configuration import Config
from flytekit.remote import FlyteRemote
from grpc._channel import _InactiveRpcError

from unionml import Model

FLYTECTL_CMD = "sandbox" if os.getenv("UNIONML_CI", False) else "demo"
NO_CLUSTER_MSG = "🛑 no demo cluster found" if FLYTECTL_CMD == "demo" else "🛑 no Sandbox found"
RETRY_ERROR = "failed to create workflow in propeller namespaces"


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
        subprocess.run(["flytectl", FLYTECTL_CMD, "start", "--source", "."])

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
    if spec is None:
        raise TypeError(f"module {module_name} couldn't be loaded from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if spec.loader is None:
        raise AttributeError(f"spec {spec} doesn't have a defined loader.")
    spec.loader.exec_module(module)
    return module.model


def _retry_execution(fn, n_retries: int = 100, wait_time: int = 3):
    for _ in range(n_retries):
        try:
            return fn()
        except _InactiveRpcError as exc:
            if not exc.details().startswith(RETRY_ERROR):
                raise
            time.sleep(wait_time)


@pytest.mark.parametrize(
    "ml_framework, hyperparameters, trainer_kwargs",
    [
        ["sklearn", {"C": 1.0, "max_iter": 1000}, {}],
        [
            "pytorch",
            {"in_dims": 64, "hidden_dims": 32, "out_dims": 10},
            {"batch_size": 512, "n_epochs": 10, "learning_rate": 0.0003},
        ],
        [
            "keras",
            {"in_dims": 64, "hidden_dims": 32, "out_dims": 10},
            {"batch_size": 512, "n_epochs": 100, "learning_rate": 0.0003},
        ],
    ],
    ids=["sklearn", "pytorch", "keras"],
)
def test_unionml_deployment(
    flyte_remote: FlyteRemote,
    ml_framework: str,
    hyperparameters: Dict[str, Any],
    trainer_kwargs: Dict[str, Any],
):
    model = _import_model_from_file(
        f"tests.integration.{ml_framework}_app.quickstart",
        Path(__file__).parent / f"{ml_framework}_app" / "quickstart.py",
    )
    project = "unionml-integration-tests"
    model.name = f"{model.name}-{ml_framework}"
    model.remote(
        dockerfile="Dockerfile.test",
        project=project,
        domain="development",
    )
    app_version = str(uuid.uuid4())

    model.remote_deploy(app_version=app_version)

    # this is a hack to account for lag between project and propeller namespace creation
    model_artifact = _retry_execution(
        lambda: model.remote_train(
            app_version=app_version,
            wait=True,
            hyperparameters=hyperparameters,
            trainer_kwargs=trainer_kwargs,
        )
    )
    assert isinstance(model_artifact.model_object, model.model_type)
    assert isinstance(model_artifact.metrics, dict)
    assert isinstance(model_artifact.metrics["test"], float)
    assert isinstance(model_artifact.metrics["train"], float)
