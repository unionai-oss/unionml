# flytekit-learn

The easiest way to build and deploy models

# Local

Run training and prediction locally

```bash
python example/workflows/app.py
```

Run FastAPI app:

```bash
uvicorn example.app.main:app --reload
```

In a different shell session, make calls to the API that run locally

```bash
python example/requests/local.py
```

Or using `curl`:

```bash
source example/requests/local.sh
```

# Sandbox

Start the sandbox cluster

```bash
flytectl sandbox start --source .
```

Build the app image in the sandbox container

```bash
flytectl sandbox exec -- docker build . --tag "flytekit-learn:v0"
```

Deploy the model to Flyte backend

```bash
fklearn deploy example.app.main:model -i "flytekit-learn:v0" -v 0
```

Train model on a Flyte backend

```bash
fklearn train example.app.main:model -i '{"hyperparameters": {"C": 1.0, "max_iter": 1000}, "sample_frac": 1.0, "random_state": 123}'
```

Generate predictions from reader:

```bash
fklearn predict example.app.main:model -i '{"sample_frac": 0.01, "random_state": 123}'
```

Generate predictions from with feature data:

```bash
fklearn predict example.app.main:model -f example/data/sample_breast_cancer_data.json
```

Call the Flyte backend via the FastAPI app:

```bash
python example/requests/remote.py
```

Or using `curl`:

```bash
source example/requests/remote.sh
```

# Remote

## Build and Push Docker Image

```bash
export FLYTE_CONFIG=config/remote.config
./docker_build_and_tag.sh
```

## Start FastAPI Server Locally

```bash
export FLYTE_CONFIG=config/remote.config
uvicorn app.workflows.app:app --reload
```

**Note:** Make sure to replace the `client_secret` entry in `config/remote.config`

Deploy the model to remote Flyte backend

```bash
fklearn deploy example.app.main:model -i "ghcr.io/unionai-oss/flytekit-learn:$VERSION" -v $VERSION
```

To run CLI commands or FastAPI endpoint calls against the remote Flyte backend, you'll need
to have the correct credentials in your `~/.aws/credentials` file, or you can export the following aws
credentials to you environment:

```
export AWS_ACCESS_KEY_ID="<REPLACE_WITH_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<REPLACE_WITH_SECRET_KEY>"
export AWS_SESSION_TOKEN="<REPLACE_WITH_SESSION_TOKEN>"
```

Train model on a Flyte backend

```bash
fklearn train example.app.main:model -i '{"hyperparameters": {"C": 1.0, "max_iter": 1000}, "sample_frac": 1.0, "random_state": 123}'
```

Generate predictions from reader:

```bash
fklearn predict example.app.main:model -i '{"sample_frac": 0.01, "random_state": 123}'
```

Generate predictions from with feature data:

```bash
fklearn predict example.app.main:model -f example/data/sample_breast_cancer_data.json
```

Call the Flyte backend via the FastAPI app:
