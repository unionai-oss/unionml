(deploying_to_production)=

# Deployment Guide

```{toctree}
---
hidden: true
---
flyte_cluster
serving_fastapi
serving_aws_lambda
```

Currently, UnionML apps support two core classes of microservice: model training and model serving.

## Training Models

`unionml` uses `flytekit` under the hood to execute your training workflows locally, but you
can benefit from the reprodicubility and scalability benefits of `unionml` by deploying your workflows
to a production-grade `flyte` cluster.

- {ref}`Flyte Cluster<flyte_cluster>`: Deploy training and prediction services to a Flyte cluster.

## Serving Predictions

Once you have a trained model object that you want to serve in production, you can deploy a
prediction endpoint using FastAPI or AWS Lambda functions:

- {ref}`Serving with FastAPI <serving_fastapi>`: Stand up a prediction microservice with FastAPI.
- {ref}`Serving with AWS Lambda <serving_aws_lambda>`: Create a serverless endpoint with AWS Lambda.
