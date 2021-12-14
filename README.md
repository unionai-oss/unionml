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

## Build Docker Image

```bash
docker build -t ghcr.io/unionai-oss/flytekit-learn:latest .
```

## Push

```bash
docker push ghcr.io/unionai-oss/flytekit-learn:latest
```

## Start FastAPI Server Locally

```bash
export FLYTE_CONFIG=config/remote.config
uvicorn app.workflows.app:app --reload
```

**Note:** Make sure to replace the `client_secret` entry in `config/remote.config`

Deploy the model to remote Flyte backend

```bash
fklearn deploy example.app.main:model -i "ghcr.io/unionai-oss/flytekit-learn:latest" -v 0
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
