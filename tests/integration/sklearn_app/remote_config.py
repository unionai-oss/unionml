from pathlib import Path

from tests.integration.sklearn_app.quickstart import model

model.remote(
    dockerfile="Dockerfile",
    config_file=str(Path.home() / ".flyte" / "config.yaml"),
    project="digits-classifier",
    domain="development",
)
