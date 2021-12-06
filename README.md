# flytekit-learn

The easiest way to build and deploy models

# Local

```bash
flytectl sandbox start --source .
flytectl sandbox exec -- docker build . --tag "flytekit-learn:v0"
```

Run training and prediction locally

```bash
python 
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
