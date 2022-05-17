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

Currently, UnionML apps support two core types of machine learning microservices: model training and
model serving.

## Production Backend Deployment

`unionml` uses `flytekit` under the hood to execute your training and prediction workflows locally, but you
can benefit from the reproducibility and scalability benefits of `unionml` by deploying your workflows
to a production-grade `flyte` cluster.

- {ref}`Deploy to a Flyte Cluster<flyte_cluster>`: Deploy training and prediction services to a Flyte cluster.

## Serving Predictions

Once you have a trained model object that you want to serve in production, you can:

- {ref}`Serve with FastAPI <serving_fastapi>`: Stand up an online prediction service with FastAPI.
- {ref}`Serve with AWS Lambda <serving_aws_lambda>`: Create an online prediction serverless endpoint with AWS Lambda.
