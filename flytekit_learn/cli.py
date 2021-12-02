import importlib
from dataclasses import asdict

import typer

import flytekit_learn
from flytekit import LaunchPlan
from flytekit.remote import FlyteRemote


app = typer.Typer()


@app.command()
def deploy(
    app: str,
    config_path: str = typer.Option(..., "--config-path", "-c", help="path to flytekit configuration file"),
    project: str = typer.Option(..., "--project", "-p", help="project name"),
    domain: str = typer.Option(..., "--domain", "-d", help="domain name"),
    version: str = typer.Option(..., "--version", "-v", help="version"),
):
    typer.echo(f"Deploying {app}")
    remote = FlyteRemote.from_config(
        config_file_path=config_path,
        default_project=project,
        default_domain=domain,
    )
    module_name, model_var = app.split(":")
    module = importlib.import_module(module_name)
    model = getattr(module, model_var)
    train_wf = model.train(lazy=True)
    identifiers = asdict(remote._resolve_identifier_kwargs(train_wf, project, domain, train_wf.name, version))
    remote._register_entity_if_not_exists(train_wf, identifiers)
    remote.register(train_wf, **identifiers)
    train_lp = LaunchPlan.get_or_create(train_wf)
    remote.register(train_lp, **identifiers)


@app.command()
def train():
    typer.echo("Train")


if __name__ == "__main__":
    app()
