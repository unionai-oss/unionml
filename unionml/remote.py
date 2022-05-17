"""Module for flyte remote helper functions."""

import importlib
import logging
import typing

import docker
import git
from flytekit.configuration import ImageConfig, SerializationSettings
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytekit.models.project import Project
from flytekit.remote import FlyteRemote, FlyteWorkflowExecution

from unionml.model import Model, ModelArtifact

IMAGE_NAME = "unionml"
FLYTE_SANDBOX_CONTAINER_NAME = "flyte-sandbox"


logger = logging.getLogger("unionml")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logger.addHandler(handler)


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


def get_app_version() -> str:
    repo = git.Repo(".", search_parent_directories=True)
    commit = repo.rev_parse("HEAD")
    return commit.hexsha


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

    _, build_logs = sandbox_container.exec_run(["docker", "build", "/root", "--tag", image_fqn], stream=True)
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


def deploy_wf(wf, remote: FlyteRemote, image: str, project: str, domain: str, version: str):
    """Register all tasks, workflows, and launchplans needed to execute the workflow."""
    logger.info(f"Deploying workflow {wf.name}")
    serialization_settings = SerializationSettings(
        project=project,
        domain=domain,
        image_config=ImageConfig.auto(img_name=image),
    )
    remote.register_workflow(wf, serialization_settings, version)


def get_model_execution(
    model: Model,
    app_version: typing.Optional[str] = None,
    model_version: typing.Optional[str] = "latest",
) -> FlyteWorkflowExecution:
    if model._remote is None:
        raise RuntimeError("You need to configure the remote client with the `Model.remote` method")

    app_version = app_version or get_app_version()
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
                filters.Equal("launch_plan.name", train_wf.id.name),
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

    app_version = app_version or get_app_version()
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
            filters.Equal("launch_plan.name", train_wf.id.name),
            filters.Equal("phase", "SUCCEEDED"),
        ],
        sort_by=Sort("created_at", Sort.Direction.DESCENDING),
    )
    return [x.id.name for x in executions]
