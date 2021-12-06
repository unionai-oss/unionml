# flytekit-learn

The easiest way to build and deploy models

# Local

Run training and prediction locally

```bash
python example/workflows/app.py
```

Run FastAPI app:

```bash
uvicorn example.workflows.app:app --reload
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
fklearn deploy example.workflows.app:model -i "flytekit-learn:v0"
```

Train model on a Flyte backend

```bash
fklearn train example.workflows.app:model -h '{"C": 1.0, "max_iter": 1000}'
```

Generate predictions:

```bash
fklearn predict example.workflows.app:model -f example/data/sample_breast_cancer_data.json
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
docker build -t ghcr.io/flyteorg/flytekit-learn:latest .
```

## Push

```bash
docker push ghcr.io/flyteorg/flytekit-learn:latest
```

## Export Flyte Image

```bash
export FLYTE_INTERNAL_IMAGE=ghcr.io/flyteorg/flytekit-learn:latest
```

## Start FastAPI Server Locally

```bash
uvicorn app.workflows.app:app --reload
```
