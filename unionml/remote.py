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


def get_image_fqn(model: Model, version: str, image_name: typing.Optional[str] = None) -> str:
    image_name = IMAGE_NAME if image_name is None else image_name
    return f"{model.registry}/{image_name}:{model.name.replace('_', '-')}-{version}"


def sandbox_docker_build(model: Model, image_fqn: str):
    logger.info("Building docker container in flyte sandbox.")
    client = docker.from_env()
    sandbox_container = None
    for container in client.containers.list():
        if container.name == FLYTE_SANDBOX_CONTAINER_NAME:
            sandbox_container = container

    if sandbox_container is None:
        raise RuntimeError(
            "Cannot find Flyte Sandbox. Make sure to install flytectl and create a sandbox with "
            "`flytectl sandbox start --source .`"
        )

    _, build_logs = sandbox_container.exec_run(["docker", "build", "/root", "--tag", image_fqn])
    for line in build_logs.decode().splitlines():
        logger.info(line)


def docker_build_push(model: Model, image_fqn: str) -> docker.models.images.Image:
    client = docker.from_env()

    logger.info(f"Building image: {image_fqn}")
    image, build_logs = client.images.build(
        path=".",
        dockerfile=model.dockerfile,
        tag=image_fqn,
        rm=True,
    )
    for line in build_logs:
        logger.info(line)

    for line in client.api.push(image.tags[0], stream=True, decode=True):
        logger.info(line)

    return image


def deploy_wf(wf, remote: FlyteRemote, image: str, project: str, domain: str, version: str):
    """Register all tasks, workflows, and launchplans needed to execute the workflow."""
    logger.info(f"Deploying workflow {wf.name}")
    serialization_settings = SerializationSettings(
        project=project,
        domain=domain,
        image_config=ImageConfig.auto(img_name=image),
    )
    remote.register_workflow(wf, serialization_settings, version)


def get_latest_model_artifact(model: Model, app_version: typing.Optional[str] = None) -> ModelArtifact:
    if model._remote is None:
        raise RuntimeError("You need to configure the remote client with the `Model.remote` method")

    app_version = get_app_version()
    train_wf = model._remote.fetch_workflow(
        model._remote._default_project,
        model._remote._default_domain,
        model.train_workflow_name,
        app_version,
    )
    [latest_training_execution, *_], _ = model._remote.client.list_executions_paginated(
        train_wf.id.project,
        train_wf.id.domain,
        limit=1,
        filters=[
            filters.Equal("launch_plan.name", train_wf.id.name),
            filters.Equal("phase", "SUCCEEDED"),
        ],
        sort_by=Sort("created_at", Sort.Direction.DESCENDING),
    )
    latest_training_execution = FlyteWorkflowExecution.promote_from_model(latest_training_execution)
    model._remote.sync(latest_training_execution)

    return ModelArtifact(
        latest_training_execution.outputs["trained_model"],
        latest_training_execution.outputs["metrics"],
    )
