import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Type

import pytest
from flytekit.configuration import Config
from flytekit.remote import FlyteRemote
from grpc._channel import _InactiveRpcError

from unionml import Model
from unionml.model import ModelArtifact

FLYTECTL_CMD = "sandbox" if os.getenv("UNIONML_CI", False) else "demo"
NO_CLUSTER_MSG = "🛑 no demo cluster found" if FLYTECTL_CMD == "demo" else "🛑 no Sandbox found"
RETRY_ERROR = "failed to create workflow in propeller namespaces"


def _wait_for_flyte_cluster(remote: FlyteRemote, max_retries: int = 300, wait: int = 3):
    for _ in range(30):
        try:
            projects, *_ = remote.client.list_projects_paginated(limit=5, token=None)
            return projects
        except Exception:
            time.sleep(wait)

    raise TimeoutError("timeout expired waiting for Flyte cluster to start.")


@pytest.fixture(scope="session")
def flyte_remote():
    cluster_preexists = True
    try:
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
        projects = _wait_for_flyte_cluster(remote)
        assert "flytesnacks" in [p.id for p in projects]
        assert "flytesnacks" in [p.name for p in projects]
        yield remote
        r = subprocess.run(["kubectl", "--kubeconfig", f"{os.environ['HOME']}/.flyte/k3s/k3s.yaml", "-n", "flyte", "get", "pods"])
        print(f"{r}")
    finally:
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


def _assert_model_artifact(model_artifact: ModelArtifact, model_type: Type):
    assert isinstance(model_artifact.model_object, model_type)
    assert isinstance(model_artifact.metrics, dict)
    assert isinstance(model_artifact.metrics["test"], float)
    assert isinstance(model_artifact.metrics["train"], float)

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
    model.dataset.name = f"{model.dataset.name}-{ml_framework}"
    model.remote(
        dockerfile=f"ci/py{''.join(str(x) for x in sys.version_info[:2])}/Dockerfile",
        project=project,
        domain="development",
    )
    app_version: str = model.remote_deploy(allow_uncommitted=True)
    kwargs = {"hyperparameters": hyperparameters, "trainer_kwargs": trainer_kwargs}

    # this is a hack to account for lag between project and propeller namespace creation
    model_artifact = _retry_execution(lambda: model.remote_train(app_version=app_version, wait=True, **kwargs))
    _assert_model_artifact(model_artifact, model.model_type)

    # test patch deployment
    patch_app_version = model.remote_deploy(patch=True)

    # the default (latest) workflow version should be the same as explicitly passing in the patch app version
    execution_latest = _retry_execution(lambda: model.remote_train(wait=False, **kwargs))  # type: ignore
    execution_patch_explicit = _retry_execution(
        lambda: model.remote_train(app_version=patch_app_version, wait=False, **kwargs)
    )
    assert execution_latest.spec.launch_plan.version == patch_app_version
    assert execution_patch_explicit.spec.launch_plan.version == patch_app_version

    model.remote_load(execution_latest)
    _assert_model_artifact(model.artifact, model.model_type)  # type: ignore

    model.remote_load(execution_patch_explicit)
    _assert_model_artifact(model.artifact, model.model_type)  # type: ignore
