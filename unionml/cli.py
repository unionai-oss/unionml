"""unionml cli."""

import copy
import json
import os
import sys
import typing
from enum import Enum
from pathlib import Path

import click
import typer
import uvicorn
from cookiecutter.main import cookiecutter

from unionml.remote import VersionFetchError, get_app_version, get_model, get_model_execution, get_prediction_execution

sys.path.append(os.curdir)

app = typer.Typer()


IMAGE_PREFIX = "unionml"
FLYTE_SANDBOX_CONTAINER_NAME = "flyte-sandbox"


class AppTemplate(str, Enum):
    basic = "basic"
    basic_aws_lambda = "basic-aws-lambda"
    basic_aws_lambda_s3 = "basic-aws-lambda-s3"
    basic_bentoml = "basic-bentoml"
    quickdraw = "quickdraw"


class PredictionOutputFormats(str, Enum):
    json = "json"


@app.command()
def init(
    app_name: str,
    template: AppTemplate = typer.Option(
        AppTemplate.basic,
        "--template",
        "-t",
        help="template to use for initializing your app.",
    ),
):
    r"""Initialize a UnionML project."""
    config = {
        "app_name": app_name,
    }
    cookiecutter(
        str(Path(__file__).parent / "templates" / template.value),
        no_input=True,
        extra_context=config,
    )


@app.command()
def deploy(
    app: str,
    allow_uncommitted: bool = typer.Option(
        False,
        "--allow-uncommitted",
        help="Deploy uncommitted changes in the unionml project",
    ),
    patch: bool = typer.Option(
        False,
        "--patch",
        help="Bypass Docker build process and update the UnionML app source code using the latest available image.",
    ),
    schedule: bool = typer.Option(
        True,
        "--schedule/--no-schedule",
        help="Indicates whether or not to deploy the training and prediction schedules.",
    ),
):
    """Deploy model to a Flyte backend."""
    typer.echo(f"[unionml] deploying {app}")
    model = get_model(app)
    try:
        model.remote_deploy(allow_uncommitted=allow_uncommitted, patch=patch, schedule=schedule)
    except VersionFetchError as e:
        typer.echo(f"[unionml] failed to get app version: {e}", err=True)
        typer.echo(
            "[unionml] Please commit your changes or explicitly ignore this using the --allow-uncommitted flag.",
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[unionml] failed to deploy {app}: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def activate_schedules(
    app: str,
    app_version: str = typer.Option(None, "--app-version", "-v", help="App version"),
    schedule_names: typing.List[str] = typer.Option(
        None,
        "--name",
        help="Name of the schedule to activate. This option can be specified multiple times.",
    ),
):
    """Activate training and prediction schedules specified in your UnionML app."""
    typer.echo(f"[unionml] activating schedules {schedule_names} for {app}")
    model = get_model(app)
    model.remote_activate_schedules(app_version=app_version, schedule_names=schedule_names)


@app.command()
def deactivate_schedules(
    app: str,
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    schedule_names: typing.List[str] = typer.Option(
        None,
        "--name",
        help="Name of the schedule to deactivate. This option can be specified multiple times.",
    ),
):
    """Deactivate training and prediction schedules specified in your UnionML app."""
    typer.echo(f"[unionml] deactivating schedules {schedule_names} for {app}")
    model = get_model(app)
    model.remote_activate_schedules(app_version=app_version, schedule_names=schedule_names)


@app.command()
def train(
    app: str,
    inputs: str = typer.Option(None, "--inputs", "-i", help="json string of inputs to pass into training workflow"),
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    wait: bool = typer.Option(False, "--wait", "-w", help="whether or not to wait for remote execution to complete."),
):
    r"""Train a model."""
    typer.echo(f"[unionml] app: {app} - training model")
    model = get_model(app)
    train_inputs = {}
    if inputs:
        train_inputs.update(json.loads(inputs))
    model.remote_train(app_version, wait, **train_inputs)
    if wait:
        assert model.artifact is not None
        typer.echo("[unionml] training completed with model artifacts:")
        typer.echo(f"[unionml] model object: {model.artifact.model_object}")
        typer.echo(f"[unionml] model metrics: {model.artifact.metrics}")


@app.command()
def predict(
    app: str,
    inputs: str = typer.Option(None, "--inputs", "-i", help="json string of inputs tp pass into predict workflow"),
    features: Path = typer.Option(None, "--features", "-f", help="generate predictions for this feature"),
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    model_version: str = typer.Option(None, "--model-version", "-m", help="model version"),
    wait: bool = typer.Option(False, "--wait", "-w", help="whether or not to wait for remote execution to complete."),
):
    r"""Generate prediction."""
    typer.echo(f"[unionml] app: {app} - generating predictions")
    model = get_model(app)

    prediction_inputs = {}
    if inputs:
        prediction_inputs.update(json.loads(inputs))
    elif features:
        prediction_inputs.update({"features": model._dataset.get_features(features)})

    predictions = model.remote_predict(app_version, model_version, wait=wait, **prediction_inputs)
    if wait:
        typer.echo(f"[unionml] predictions: {predictions}")


@app.command("list-model-versions")
def list_model_versions(
    app: str,
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of model versions to list, sorted in descending order of time of execution."
    ),
):
    r"""List all model versions."""
    model = get_model(app)
    app_version = app_version or get_app_version(allow_uncommitted=True)
    typer.echo(f"[unionml] app: {app} - listing model versions for app version={app_version}")
    for model_version in model.remote_list_model_versions(app_version, limit):
        typer.echo(f"- {model_version}")


@app.command("list-prediction-ids")
def list_prediction_ids(
    app: str,
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of model versions to list, sorted in descending order of time of execution."
    ),
):
    r"""List all batch prediction identifiers."""
    model = get_model(app)
    app_version = app_version or get_app_version(allow_uncommitted=True)
    typer.echo(f"[unionml] app: {app} - listing prediction ids for app version={app_version}")
    for prediction_id in model.remote_list_prediction_ids(app_version, limit):
        typer.echo(f"- {prediction_id}")


@app.command()
def list_scheduled_training_runs(
    app: str,
    schedule_name,
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    limit: int = typer.Option(5, "--limit", help="number of "),
):
    r"""List scheduled training runs by schedule name."""
    model = get_model(app)
    app_version = app_version or get_app_version(allow_uncommitted=True)
    typer.echo(f"[unionml] app: {app} - listing scheduled training runs for schedule '{schedule_name}'")
    for run in model.remote_list_scheduled_training_runs(schedule_name, app_version=app_version, limit=limit):
        typer.echo(f"- {run.id.name}")


@app.command()
def list_scheduled_prediction_runs(
    app: str,
    schedule_name,
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    limit: int = typer.Option(5, "--limit", help="number of "),
):
    r"""List scheduled prediction runs by schedule name."""
    model = get_model(app)
    app_version = app_version or get_app_version(allow_uncommitted=True)
    typer.echo(f"[unionml] app: {app} - listing scheduled training runs for schedule '{schedule_name}'")
    for run in model.remote_list_scheduled_training_runs(schedule_name, app_version=app_version, limit=limit):
        typer.echo(f"- {run.id.name}")


@app.command("fetch-model")
def fetch_model(
    app: str,
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    model_version: str = typer.Option("latest", "--model-version", "-m", help="model version"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="output file path"),
    kwargs: str = typer.Option(None, "--kwargs", help="json string of kwargs to pass into model.save"),
):
    r"""Fetch a model object from the remote backend."""
    model = get_model(app)
    app_version = app_version or get_app_version(allow_uncommitted=True)
    execution = get_model_execution(model, app_version, model_version=model_version)
    model.remote_load(execution)
    save_kwargs = {}
    if kwargs is not None:
        save_kwargs = json.loads(kwargs)
    model.save(output_file, **save_kwargs)
    typer.echo(f"[unionml] app: {app} - saving model version {execution.id.name} to {output_file}")


@app.command("fetch-predictions")
def fetch_predictions(
    app: str,
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
    prediction_id: str = typer.Option("latest", "--prediction-id", "-m", help="prediction id"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="output file path"),
    output_format: str = typer.Option(
        PredictionOutputFormats.json,
        "--output-format",
        "-f",
        help="Output format of the file. Currently only json-serializable prediction outputs are supported.",
    ),
):
    r"""Fetch a model object from the remote backend."""
    model = get_model(app)
    app_version = app_version or get_app_version(allow_uncommitted=True)
    execution = get_prediction_execution(model, app_version, prediction_id=prediction_id)
    predictions = model.remote_fetch_predictions(execution)
    if output_format == "json":
        with open(output_file) as f:
            json.dump(predictions, f)
    else:
        raise ValueError(f"output_format '{output_format}' not recognized.")
    typer.echo(f"[unionml] app: {app} - saving predictions {execution.id.name} to {output_file}")


@app.callback()
def callback():
    r"""unionml command-line tool."""


def serve_command():
    r"""Modify the uvicorn.main entrypoint for unionml app serving."""
    fn = copy.deepcopy(uvicorn.main)
    fn.name = "serve"
    fn.short_help = "Serve an unionml model."
    fn.help = (
        "Serve an unionml model using uvicorn. This command uses the main uvicorn entrypoint with an additional"
        " ``--model-path`` argument.\n\nFor more information see: https://www.uvicorn.org/#command-line-options"
    )

    option = click.Option(param_decls=["--model-path"], default=None, help="model path to use for serving", type=Path)
    fn.params.append(option)

    callback = fn.callback

    def custom_callback(**kwargs):
        if os.getenv("UNIONML_MODEL_PATH"):
            typer.echo(
                f"UNIONML_MODEL_PATH environment variable is set to {os.getenv('UNIONML_MODEL_PATH')}. "
                "Please unset this variable before running ``unionml serve``.",
                err=True,
            )
            raise typer.Exit(code=1)

        model_path = kwargs.pop("model_path")
        if model_path is not None:
            if not model_path.exists():
                typer.echo(f"Model path {model_path} not found.", err=True)
                raise typer.Exit(code=1)
            os.environ["UNIONML_MODEL_PATH"] = str(model_path)
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
