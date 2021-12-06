"""flytekit-learn cli."""

import importlib
import json
import os
import typing
from dataclasses import asdict
from inspect import signature
from pathlib import Path

import typer

import flytekit_learn
from flytekit import LaunchPlan
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytekit.remote import FlyteRemote, FlyteWorkflowExecution


app = typer.Typer()


def _get_model(app: str):
    module_name, model_var = app.split(":")
    return getattr(importlib.import_module(module_name), model_var)


def _deploy_wf(wf, remote, project, domain, version):
    """Register all tasks, workflows, and launchplans needed to execute the workflow."""
    lp = LaunchPlan.get_or_create(wf)
    identifiers = asdict(remote._resolve_identifier_kwargs(wf, project, domain, wf.name, version))
    remote._register_entity_if_not_exists(wf, identifiers)
    remote.register(wf, **identifiers)
    remote.register(lp, **identifiers)


@app.command()
def deploy(
    app: str,
    image: typing.Optional[str] = typer.Option(None, "--image", "-i", help="default image to use"),
    project: str = typer.Option(None, "--project", "-p", help="project name"),
    domain: str = typer.Option(None, "--domain", "-d", help="domain name"),
    name: typing.Optional[str] = typer.Option(None, "--name", "-n", help="model name"),
    version: str = typer.Option(None, "--version", "-v", help="version"),
):
    """Deploy model to a Flyte backend."""

    typer.echo(f"[fklearn] deploying {app}")
    os.environ["FLYTE_INTERNAL_IMAGE"] = image
    model = _get_model(app)

    if model.name is None:
        if name is None:
            typer.echo("name must be provided in the flytekit_learn.Model constructor or the --name option", err=True)
        model.name = name

    # get training workflow
    train_wf = model.train(lazy=True)
    predict_wf = model.predict(lazy=True)

    # register all tasks, workflows, and launchplans needed to execute model endpoints
    args = [project, domain, version]
    _deploy_wf(train_wf, model._remote, *args)
    _deploy_wf(predict_wf, model._remote, *args)


@app.command()
def train(
    app: str,
    hyperparameters: str = typer.Option(None, "--hyperparameters", "-h", help="hyperparameters"),
    project: str = typer.Option(None, "--project", "-p", help="project name"),
    domain: str = typer.Option(None, "--domain", "-d", help="domain name"),
    name: typing.Optional[str] = typer.Option(None, "--name", "-n", help="model name"),
    version: str = typer.Option(None, "--version", "-v", help="version"),
):
    typer.echo(f"[fklearn] app: {app} - training model")
    model = _get_model(app)

    if model.name is None:
        if name is None:
            typer.echo("name must be provided in the flytekit_learn.Model constructor or the --name option", err=True)
        model.name = name

    hyperparameters = json.loads(hyperparameters)

    train_wf = model._remote.fetch_workflow(project, domain, model.train_workflow_name, version)
    typer.echo(f"[fklearn] executing model workflow")
    typer.echo(f"[fklearn] project: {train_wf.id.project}")
    typer.echo(f"[fklearn] domain: {train_wf.id.domain}")
    typer.echo(f"[fklearn] name: {train_wf.id.name}")
    typer.echo(f"[fklearn] version: {train_wf.id.version}")
    typer.echo(f"[fklearn] hyperparameters: {hyperparameters}")

    execution = model._remote.execute(train_wf, inputs={"hyperparameters": hyperparameters}, wait=True)
    typer.echo(f"[fklearn] training completed with outputs:")
    for k, v in execution.outputs.items():
        typer.echo(f"[fklearn] {k}: {v}")


@app.command()
def predict(
    app: str,
    features: Path = typer.Option(None, "--features", "-f", help="hyperparameters"),
    project: str = typer.Option(None, "--project", "-p", help="project name"),
    domain: str = typer.Option(None, "--domain", "-d", help="domain name"),
    name: typing.Optional[str] = typer.Option(None, "--name", "-n", help="model name"),
    version: str = typer.Option(None, "--version", "-v", help="version"),
):
    typer.echo(f"[fklearn] app: {app} - generating predictions")
    model = _get_model(app)

    if model.name is None:
        if name is None:
            typer.echo("name must be provided in the flytekit_learn.Model constructor or the --name option", err=True)
        model.name = name

    with features.open() as f:
        features = json.load(f)

    train_wf = model._remote.fetch_workflow(project, domain, model.train_workflow_name, version)
    predict_wf = model._remote.fetch_workflow(project, domain, model.predict_workflow_name, version)
    features = model._dataset(features=features)()

    typer.echo(f"[fklearn] getting latest model")
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
    trained_model = latest_training_execution.outputs["model"]

    predictions = model._remote.execute(
        predict_wf, inputs={"model": trained_model, "features": features}, wait=True
    )
    typer.echo(f"[fklearn predictions: {predictions.outputs['o0']}")


@app.command()
def schedule():
    # TODO
    typer.echo("scheduling")


if __name__ == "__main__":
    app()
