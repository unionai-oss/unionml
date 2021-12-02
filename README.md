# flytekit-learn

The easiest way to build and deploy models

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