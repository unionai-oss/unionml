"""Module for flyte remote helper functions."""

import contextlib
import importlib
import pathlib
import typing

import docker
import git
from flytekit import LaunchPlan
from flytekit.configuration import FastSerializationSettings, ImageConfig, SerializationSettings
from flytekit.core.workflow import WorkflowBase
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytekit.models.project import Project
from flytekit.remote import FlyteRemote, FlyteWorkflowExecution
from flytekit.tools import fast_registration, repo

from unionml._logging import logger
from unionml.model import Model, ModelArtifact

IMAGE_NAME = "unionml"
FLYTE_SANDBOX_CONTAINER_NAME = "flyte-sandbox"


class VersionFetchError(RuntimeError):
    pass


def get_model(app: str, reload: bool = False) -> Model:
    module_name, model_var = app.split(":")
    module = importlib.import_module(module_name)
    if reload:
        importlib.reload(module)
    return getattr(module, model_var)


def create_project(remote: FlyteRemote, project: typing.Optional[str]):
    project = project or remote.default_project
    projects, _ = remote.client.list_projects_paginated(filters=[filters.Equal("project.name", project)])
    if not projects:
        remote.client.register_project(Project(id=project, name=project, description=project))


def get_app_version(allow_uncommitted: bool = False) -> str:
    repo = git.Repo(".", search_parent_directories=True)
    if repo.is_dirty():
        if not allow_uncommitted:
            raise VersionFetchError("Version number cannot be determined with uncommitted changes present.")
        logger.warning("You have uncommitted changes, unionml is using the the latest commit as the app version.")

    with contextlib.suppress(git.CommandError):
        if list(repo.iter_commits("@{upstream}..")):
            logger.warning(
                "You have local commits that are not present on remote repositories,"
                " which may cause issues with deployment."
            )

    return repo.head.commit.hexsha


def get_image_fqn(model: Model, app_version: str, image_name: typing.Optional[str] = None) -> str:
    image_name = IMAGE_NAME if image_name is None else image_name
    if model.registry is None:
        image_uri = image_name
    else:
        image_uri = f"{model.registry}/{image_name}"
    return f"{image_uri}:{model.name.replace('_', '-')}-{app_version}"


def sandbox_docker_build(model: Model, image_fqn: str):
    logger.info("Building docker container in flyte demo cluster.")
    client = docker.from_env()
    sandbox_container = None
    for container in client.containers.list():
        if container.name == FLYTE_SANDBOX_CONTAINER_NAME:
            sandbox_container = container

    if sandbox_container is None:
        raise RuntimeError(
            "Cannot find Flyte Demo Cluster. Make sure to install flytectl and create a sandbox with "
            "`flytectl demo start --source .`"
        )

    _, build_logs = sandbox_container.exec_run(
        ["docker", "build", "/root", "--tag", image_fqn, "--file", f"/root/{model.dockerfile}"],
        stream=True,
    )
    for line in build_logs:
        logger.info(line.decode().strip())


def docker_build_push(model: Model, image_fqn: str) -> docker.models.images.Image:
    if model.registry is None:
        raise ValueError("You must specify a registry in `model.remote` when deploying to a remote cluster.")

    client = docker.from_env()

    logger.info(f"Building image: {image_fqn}")
    build_logs = client.api.build(
        path=".",
        dockerfile=model.dockerfile,
        tag=image_fqn,
        rm=True,
    )
    for line in build_logs:
        logger.info(line.decode().strip())

    for line in client.api.push(image_fqn, stream=True, decode=True):
        logger.info(line)


def deploy_workflow(
    wf: WorkflowBase,
    remote: FlyteRemote,
    image: str,
    project: str,
    domain: str,
    version: str,
    patch: bool = False,
    patch_destination_dir: str = None,
):
    """Register all tasks, workflows, and launchplans needed to execute the workflow."""
    logger.info(f"Deploying workflow '{wf.name}'")
    fast_serialization_settings = None
    if patch:
        # Create a zip file containing all the entries.
        detected_root = repo.find_common_root(["."])
        zip_file = fast_registration.fast_package(detected_root, output_dir=None, deref_symlinks=False)

        # Upload zip file to Admin using FlyteRemote.
        _, native_url = remote._upload_file(pathlib.Path(zip_file))

        # Create serialization settings
        # TODO: Rely on default Python interpreter for now, this will break custom Spark containers
        fast_serialization_settings = FastSerializationSettings(
            enabled=True,
            destination_dir=patch_destination_dir,
            distribution_location=native_url,
        )

    serialization_settings = SerializationSettings(
        project=project,
        domain=domain,
        image_config=ImageConfig.auto(img_name=image),
        fast_serialization_settings=fast_serialization_settings,
    )

    remote.register_workflow(wf, serialization_settings, version)


def deploy_launchplan(
    lp: LaunchPlan,
    remote: FlyteRemote,
    project: str,
    domain: str,
    version: str,
    activate_on_deploy: bool = True,
):
    logger.info(f"Deploying launchplan '{lp.name}'")
    remote.register_launch_plan(lp, version=version, project=project, domain=domain)
    if activate_on_deploy:
        activate_launchplan(lp, remote, project, domain, version)


def activate_launchplan(
    lp: LaunchPlan,
    remote: FlyteRemote,
    project: str,
    domain: str,
    version: str,
):
    lp_id = remote.fetch_launch_plan(project, domain, lp.name, version).id
    remote.client.update_launch_plan(lp_id, "ACTIVE")


def deactivate_launchplan(
    lp: LaunchPlan,
    remote: FlyteRemote,
    project: str,
    domain: str,
    version: str,
):
    lp_id = remote.fetch_launch_plan(project, domain, lp.name, version).id
    remote.client.update_launch_plan(lp_id, "INACTIVE")


def get_model_execution(
    model: Model,
    app_version: typing.Optional[str] = None,
    model_version: typing.Optional[str] = "latest",
) -> FlyteWorkflowExecution:
    if model._remote is None:
        raise RuntimeError("You need to configure the remote client with the `Model.remote` method")

    train_wf = model._remote.fetch_workflow(
        model._remote._default_project,
        model._remote._default_domain,
        model.train_workflow_name,
        app_version,
    )
    if model_version is not None and model_version != "latest":
        execution = model._remote.fetch_execution(
            project=train_wf.id.project,
            domain=train_wf.id.domain,
            name=model_version,
        )
    else:
        [execution, *_], _ = model._remote.client.list_executions_paginated(
            train_wf.id.project,
            train_wf.id.domain,
            limit=1,
            filters=[
                filters.ValueIn("launch_plan.name", [train_wf.id.name] + model.training_schedule_names),
                filters.Equal("phase", "SUCCEEDED"),
            ],
            sort_by=Sort("created_at", Sort.Direction.DESCENDING),
        )
        execution = FlyteWorkflowExecution.promote_from_model(execution)
    model._remote.sync(execution)
    return execution


def get_prediction_execution(
    model: Model,
    app_version: typing.Optional[str] = None,
    prediction_id: typing.Optional[str] = "latest",
) -> FlyteWorkflowExecution:
    if model._remote is None:
        raise RuntimeError("You need to configure the remote client with the `Model.remote` method")

    predict_wf = model._remote.fetch_workflow(
        model._remote._default_project,
        model._remote._default_domain,
        model.predict_workflow_name,
        app_version,
    )
    if prediction_id is not None and prediction_id != "latest":
        execution = model._remote.fetch_execution(
            project=predict_wf.id.project,
            domain=predict_wf.id.domain,
            name=prediction_id,
        )
    else:
        [execution, *_], _ = model._remote.client.list_executions_paginated(
            predict_wf.id.project,
            predict_wf.id.domain,
            limit=1,
            filters=[
                filters.ValueIn("launch_plan.name", [predict_wf.id.name] + model.prediction_schedule_names),
                filters.Equal("phase", "SUCCEEDED"),
            ],
            sort_by=Sort("created_at", Sort.Direction.DESCENDING),
        )
        execution = FlyteWorkflowExecution.promote_from_model(execution)
    model._remote.sync(execution)
    return execution


def get_model_artifact(
    model: Model,
    app_version: typing.Optional[str] = None,
    model_version: typing.Optional[str] = "latest",
) -> ModelArtifact:
    execution = get_model_execution(model, app_version, model_version)
    model.remote_load(execution)
    assert model.artifact is not None
    return model.artifact


def list_model_versions(model: Model, app_version: typing.Optional[str] = None, limit: int = 10) -> typing.List[str]:
    if model._remote is None:
        raise RuntimeError("You need to configure the remote client with the `Model.remote` method")

    app_version = app_version or get_app_version(allow_uncommitted=True)
    train_wf = model._remote.fetch_workflow(
        model._remote._default_project,
        model._remote._default_domain,
        model.train_workflow_name,
        app_version,
    )
    executions, _ = model._remote.client.list_executions_paginated(
        train_wf.id.project,
        train_wf.id.domain,
        limit=limit,
        filters=[
            filters.ValueIn("launch_plan.name", [train_wf.id.name] + model.training_schedule_names),
            filters.Equal("launchplan.version", app_version),
            filters.Equal("phase", "SUCCEEDED"),
        ],
        sort_by=Sort("created_at", Sort.Direction.DESCENDING),
    )
    return [x.id.name for x in executions]


def list_prediction_ids(model: Model, app_version: typing.Optional[str] = None, limit: int = 10) -> typing.List[str]:
    if model._remote is None:
        raise RuntimeError("You need to configure the remote client with the `Model.remote` method")

    app_version = app_version or get_app_version(allow_uncommitted=True)
    predict_wf = model._remote.fetch_workflow(
        model._remote._default_project,
        model._remote._default_domain,
        model.predict_workflow_name,
        app_version,
    )
    executions, _ = model._remote.client.list_executions_paginated(
        predict_wf.id.project,
        predict_wf.id.domain,
        limit=limit,
        filters=[
            filters.ValueIn("launch_plan.name", [predict_wf.id.name] + model.prediction_schedule_names),
            filters.Equal("launchplan.version", app_version),
            filters.Equal("phase", "SUCCEEDED"),
        ],
        sort_by=Sort("created_at", Sort.Direction.DESCENDING),
    )
    return [x.id.name for x in executions]


def get_scheduled_runs(
    remote: FlyteRemote,
    schedule_name: str,
    app_version: typing.Optional[str] = None,
    limit: typing.Optional[int] = 100,
) -> typing.List[FlyteWorkflowExecution]:
    app_version = app_version or get_app_version(allow_uncommitted=True)
    exec_models, _ = remote.client.list_executions_paginated(
        remote.default_project,
        remote.default_domain,
        limit,
        filters=[
            filters.Equal("launch_plan.name", schedule_name),
            filters.Equal("launchplan.version", app_version),
        ],
        sort_by=Sort("created_at", Sort.Direction.DESCENDING),
    )
    return [FlyteWorkflowExecution.promote_from_model(e) for e in exec_models]
