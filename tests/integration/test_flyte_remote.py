import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Type

import pytest
from flytekit.configuration import Config
from flytekit.exceptions.user import FlyteEntityNotExistException
from flytekit.models.common import NamedEntityIdentifier
from flytekit.remote import FlyteRemote
from flytekit.remote.executions import FlyteWorkflowExecution
from grpc._channel import _InactiveRpcError

from unionml import Model
from unionml.model import ModelArtifact

FLYTECTL_CMD = "sandbox" if os.getenv("UNIONML_CI", False) else "demo"
NO_CLUSTER_MSG = "🛑 no Sandbox found"
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
            config=Config.for_sandbox(),
            default_project="flytesnacks",
            default_domain="development",
        )
        projects = _wait_for_flyte_cluster(remote)
        assert "flytesnacks" in [p.id for p in projects]
        assert "flytesnacks" in [p.name for p in projects]
        yield remote
    finally:
        if not cluster_preexists:
            # only teardown the demo cluster if it didn't preexist
            subprocess.run(["flytectl", FLYTECTL_CMD, "teardown", "--volume"])


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


def _launch_plan_is_active(remote: FlyteRemote, launch_plan_name: str):
    lp = remote.fetch_launch_plan(name=launch_plan_name)
    try:
        remote.client.get_active_launch_plan(NamedEntityIdentifier(lp.id.project, lp.id.domain, lp.id.name))
        return True
    except FlyteEntityNotExistException:
        return False


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
    if ml_framework != "sklearn":
        pytest.skip("Don't run Flyte cluster tests on other frameworks due to memory load on " "docker image in CI.")
    model = _import_model_from_file(
        f"tests.integration.{ml_framework}_app.quickstart",
        Path(__file__).parent / f"{ml_framework}_app" / "quickstart.py",
    )
    project = "unionml-integration-tests"
    model.name = f"{model.name}-{ml_framework}"
    model.dataset.name = f"{model.dataset.name}-{ml_framework}"

    # configure remote
    model.remote(
        dockerfile=f"ci/py{''.join(str(x) for x in sys.version_info[:2])}/Dockerfile",
        registry="localhost:30000",
        project=project,
        domain="development",
    )

    # schedule launchplans, which should be automatically activated with the remote_deploy() call
    training_schedule_name = f"{model.name}_training_schedule"
    prediction_schedule_name = f"{model.name}_prediction_schedule"

    app_version: str = model.remote_deploy(allow_uncommitted=True)
    kwargs = {"hyperparameters": hyperparameters, "trainer_kwargs": trainer_kwargs}

    # this is a hack to account for lag between project and propeller namespace creation
    execution: FlyteWorkflowExecution = _retry_execution(
        lambda: model.remote_train(app_version=app_version, wait=False, **kwargs)
    )
    execution = model.remote_wait(execution)
    model_artifact = model._remote_load_model_artifact(execution)
    _assert_model_artifact(model_artifact, model.model_type)

    # schedule training for patch deployment
    model.schedule_training(name=training_schedule_name, expression="*/1 * * * *", hyperparameters=hyperparameters)
    model.schedule_prediction(name=prediction_schedule_name, expression="*/1 * * * *", model_version=execution.id.name)

    # test patch deployment
    patch_app_version = model.remote_deploy(patch=True)

    assert _launch_plan_is_active(model._remote, training_schedule_name)
    assert _launch_plan_is_active(model._remote, prediction_schedule_name)

    model.remote_deactivate_schedules(patch_app_version, [training_schedule_name, prediction_schedule_name])
    assert not _launch_plan_is_active(model._remote, training_schedule_name)
    assert not _launch_plan_is_active(model._remote, prediction_schedule_name)

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
