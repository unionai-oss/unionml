"""flytekit-learn cli."""

import copy
import importlib
import json
import os
import typing
from dataclasses import asdict
from pathlib import Path

import click
import docker
import git
import typer
import uvicorn
from flytekit import LaunchPlan
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytekit.models.project import Project
from flytekit.remote import FlyteRemote, FlyteWorkflowExecution

from flytekit_learn import Model

app = typer.Typer()


IMAGE_PREFIX = "flytekit-learn"
FLYTE_SANDBOX_CONTAINER_NAME = "flyte-sandbox"


def _get_model(app: str, reload: bool = False):
    module_name, model_var = app.split(":")
    module = importlib.import_module(module_name)
    if reload:
        importlib.reload(module)
    return getattr(module, model_var)


def _create_project(remote: FlyteRemote, project: typing.Optional[str]):
    project = project or remote.default_project
    projects, _ = remote.client.list_projects_paginated(filters=[filters.Equal("project.name", project)])
    if not projects:
        typer.echo(f"Creating project {project}")
        remote.client.register_project(Project(id=project, name=project, description=project))
        return

    typer.echo(f"Using existing project {project}")


def _get_version():
    repo = git.Repo(".", search_parent_directories=True)
    if repo.is_dirty():
        typer.echo("Please commit git changes before building.", err=True)
        raise typer.Exit(code=1)
    commit = repo.rev_parse("HEAD")
    return commit.hexsha


def _get_image_fqn(model: Model, version: str):
    return f"{model.registry}/{IMAGE_PREFIX}-{model.name.replace('_', '-')}:{version}"


def _sandbox_docker_build(model: Model, image_fqn: str):
    typer.echo("Using Flyte Sandbox")
    client = docker.from_env()

    sandbox_container = None
    for container in client.containers.list():
        if container.name == FLYTE_SANDBOX_CONTAINER_NAME:
            sandbox_container = container

    if sandbox_container is None:
        typer.echo(
            "Cannot find Flyte Sandbox. Make sure to install flytectl and create a sandbox with "
            "`flytectl sandbox start --source .`",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Building image: {image_fqn}")
    _, build_logs = sandbox_container.exec_run(
        ["docker", "build", "/root", "--tag", image_fqn, "--build-arg", f"config={str(model.config_file_path)}"],
        stream=True,
    )
    for line in build_logs:
        typer.echo(line.decode())


def _docker_build_push(model: Model, image_fqn: str) -> docker.models.images.Image:
    client = docker.from_env()

    typer.echo(f"Building image: {image_fqn}")
    image, build_logs = client.images.build(
        path=".",
        dockerfile=model.dockerfile,
        tag=image_fqn,
        buildargs={
            "image": image_fqn,
            "config": str(model.config_file_path),
        },
        rm=True,
    )
    for line in build_logs:
        typer.echo(line)

    for line in client.api.push(image.tags[0], stream=True, decode=True):
        typer.echo(line)

    return image


def _deploy_wf(wf, remote: FlyteRemote, project: str, domain: str, version: str):
    """Register all tasks, workflows, and launchplans needed to execute the workflow."""
    typer.echo(f"Deploying workflow {wf.name}")
    lp = LaunchPlan.get_or_create(wf)
    identifiers = asdict(remote._resolve_identifier_kwargs(wf, project, domain, wf.name, version))
    remote._register_entity_if_not_exists(wf, identifiers)
    remote.register(wf, **identifiers)
    remote.register(lp, **identifiers)


@app.command()
def deploy(
    app: str,
    project: str = typer.Option(None, "--project", "-p", help="project name"),
    domain: str = typer.Option(None, "--domain", "-d", help="domain name"),
):
    """Deploy model to a Flyte backend."""

    typer.echo(f"[fklearn] deploying {app}")
    model = _get_model(app)

    # register all tasks, workflows, and launchplans needed to execute model endpoints
    version = _get_version()
    image = _get_image_fqn(model, version)
    args = [project, domain, version]

    # model needs to be reloaded after setting this environment variable so that the workflow's
    # default image is set correctly. This can be simplified after flytekit config improvements
    # are merged: https://github.com/flyteorg/flytekit/pull/857
    os.environ["FLYTE_INTERNAL_IMAGE"] = image or ""
    model = _get_model(app, reload=True)

    _create_project(model._remote, project)
    if model._remote._flyte_admin_url.startswith("localhost"):
        # assume that a localhost flyte_admin_url means that we want to use Flyte sandbox
        _sandbox_docker_build(model, image)
    else:
        _docker_build_push(model, image)

    for wf in [
        model.train(lazy=True),
        model.predict(lazy=True),
        model.predict(lazy=True, features=True),
    ]:
        _deploy_wf(wf, model._remote, *args)


@app.command()
def train(
    app: str,
    inputs: str = typer.Option(None, "--inputs", "-i", help="inputs to pass into training workflow"),
    project: str = typer.Option(None, "--project", "-p", help="project name"),
    domain: str = typer.Option(None, "--domain", "-d", help="domain name"),
    version: str = typer.Option(None, "--version", "-v", help="version"),
):
    """Train a model."""
    typer.echo(f"[fklearn] app: {app} - training model")
    model = _get_model(app)
    inputs = json.loads(inputs)
    version = version or _get_version()
    train_wf = model._remote.fetch_workflow(project, domain, model.train_workflow_name, version)
    typer.echo("[fklearn] executing model workflow")
    typer.echo(f"[fklearn] project: {train_wf.id.project}")
    typer.echo(f"[fklearn] domain: {train_wf.id.domain}")
    typer.echo(f"[fklearn] name: {train_wf.id.name}")
    typer.echo(f"[fklearn] version: {train_wf.id.version}")
    typer.echo(f"[fklearn] inputs: {inputs}")

    execution = model._remote.execute(train_wf, inputs=inputs, wait=True)
    typer.echo("[fklearn] training completed with outputs:")
    for k, v in execution.outputs.items():
        typer.echo(f"[fklearn] {k}: {v}")


@app.command()
def predict(
    app: str,
    inputs: str = typer.Option(None, "--inputs", "-i", help="inputs"),
    features: Path = typer.Option(None, "--features", "-f", help="hyperparameters"),
    project: str = typer.Option(None, "--project", "-p", help="project name"),
    domain: str = typer.Option(None, "--domain", "-d", help="domain name"),
    version: str = typer.Option(None, "--version", "-v", help="version"),
):
    """Generate prediction."""
    typer.echo(f"[fklearn] app: {app} - generating predictions")
    model = _get_model(app)
    version = version or _get_version()
    train_wf = model._remote.fetch_workflow(project, domain, model.train_workflow_name, version)

    typer.echo("[fklearn] getting latest model")
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
    trained_model = latest_training_execution.outputs["trained_model"]

    workflow_inputs = {"model": trained_model}
    if inputs:
        workflow_inputs.update(json.loads(inputs))
        predict_wf = model._remote.fetch_workflow(project, domain, model.predict_workflow_name, version)
    elif features:
        with features.open() as f:
            features = json.load(f)
        features = model._dataset.get_features(features)
        workflow_inputs["features"] = features
        predict_wf = model._remote.fetch_workflow(project, domain, model.predict_from_features_workflow_name, version)

    predictions = model._remote.execute(predict_wf, inputs=workflow_inputs, wait=True)
    typer.echo(f"[fklearn] predictions: {predictions.outputs['o0']}")


@app.callback()
def callback():
    """fklearn command-line tool."""


def serve_command():
    """Modify the uvicorn.main entrypoint for fklearn app serving."""
    fn = copy.deepcopy(uvicorn.main)
    fn.short_help = "Serve an fklearn model."
    fn.help = (
        "Serve an fklearn model using uvicorn. This command uses the main uvicorn entrypoint with an additional "
        "--model-path argument.\n\nFor more information see: https://www.uvicorn.org/#command-line-options"
    )

    option = click.Option(param_decls=["--model-path"], default=None, help="model path to use for serving", type=Path)
    fn.params.append(option)

    callback = fn.callback

    def custom_callback(**kwargs):
        model_path = kwargs.pop("model_path")
        if model_path is not None:
            if not model_path.exists():
                typer.echo(f"Model path {model_path} not found.", err=True)
                raise typer.Exit(code=1)
            os.environ["FKLEARN_MODEL_PATH"] = str(model_path)
        return callback(**kwargs)

    fn.callback = custom_callback
    return fn


# convert typer app to click object to define a "serve" command
# that uses the uvicorn.main entrypoint under the hood:
# https://typer.tiangolo.com/tutorial/using-click/#combine-click-and-typer
app = typer.main.get_command(app)
app.add_command(serve_command(), name="serve")


if __name__ == "__main__":
    app()
