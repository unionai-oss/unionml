from pathlib import Path

from tests.integration.sklearn_app.quickstart import model

model.remote(
    dockerfile="Dockerfile",  # points to the app's associated Dockerfile we just created
    config_file_path=str(Path.home() / ".flyte" / "config.yaml"),
    project="digits-classifier",
    domain="development",
)
