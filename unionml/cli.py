"""unionml cli."""

import copy
import json
import os
import sys
from pathlib import Path

import click
import typer
import uvicorn

from unionml.remote import get_model

sys.path.append(os.curdir)

app = typer.Typer()


IMAGE_PREFIX = "unionml"
FLYTE_SANDBOX_CONTAINER_NAME = "flyte-sandbox"


@app.command()
def deploy(app: str):
    """Deploy model to a Flyte backend."""
    typer.echo(f"[unionml] deploying {app}")
    model = get_model(app)
    model.remote_deploy()


@app.command()
def train(
    app: str,
    inputs: str = typer.Option(None, "--inputs", "-i", help="inputs to pass into training workflow"),
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
):
    """Train a model."""
    typer.echo(f"[unionml] app: {app} - training model")
    model = get_model(app)
    train_inputs = {}
    if inputs:
        train_inputs.update(json.loads(inputs))
    model_artifact = model.remote_train(app_version, **train_inputs)
    typer.echo("[unionml] training completed with model artifacts:")
    typer.echo(f"[unionml] model object: {model_artifact.object}")
    typer.echo(f"[unionml] model metrics: {model_artifact.metrics}")


@app.command()
def predict(
    app: str,
    inputs: str = typer.Option(None, "--inputs", "-i", help="inputs"),
    features: Path = typer.Option(None, "--features", "-f", help="hyperparameters"),
    app_version: str = typer.Option(None, "--app-version", "-v", help="app version"),
):
    """Generate prediction."""
    typer.echo(f"[unionml] app: {app} - generating predictions")
    model = get_model(app)

    prediction_inputs = {}
    if inputs:
        prediction_inputs.update(json.loads(inputs))
    elif features:
        with features.open() as f:
            features = json.load(f)
        prediction_inputs.update({"features": model._dataset.get_features(features)})

    predictions = model.remote_predict(app_version, **prediction_inputs)
    typer.echo(f"[unionml] predictions: {predictions}")


@app.callback()
def callback():
    """unionml command-line tool."""


def serve_command():
    """Modify the uvicorn.main entrypoint for unionml app serving."""
    fn = copy.deepcopy(uvicorn.main)
    fn.short_help = "Serve an unionml model."
    fn.help = (
        "Serve an unionml model using uvicorn. This command uses the main uvicorn entrypoint with an additional "
        "--model-path argument.\n\nFor more information see: https://www.uvicorn.org/#command-line-options"
    )

    option = click.Option(param_decls=["--model-path"], default=None, help="model path to use for serving", type=Path)
    fn.params.append(option)

    callback = fn.callback

    def custom_callback(**kwargs):
        if os.getenv("unionml_MODEL_PATH"):
            typer.echo(
                f"unionml_MODEL_PATH environment variable is set to {os.getenv('unionml_MODEL_PATH')}. "
                "Please unset this variable before running `unionml serve`.",
                err=True,
            )
            raise typer.Exit(code=1)

        model_path = kwargs.pop("model_path")
        if model_path is not None:
            if not model_path.exists():
                typer.echo(f"Model path {model_path} not found.", err=True)
                raise typer.Exit(code=1)
            os.environ["unionml_MODEL_PATH"] = str(model_path)
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
