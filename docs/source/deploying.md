(deploying_to_production)=

# Deployment Guide

```{toctree}
---
hidden: true
---
flyte_demo
serving_aws_lambda
```

Currently, UnionML apps support two core classes of microservice: model training and model serving.

## Model Training

`unionml` uses `flytekit` under the hood to execute your training workflows locally, but you
can benefit from the reprodicubility and scalability benefits of `unionml` by deploying your workflows
to a production-grade `flyte` cluster.

- {ref}`Flyte Demo Cluster<flyte_demo>`: To get a feel for how deployment works, you can deploy your
  app workflows to a local Flyte demo cluster.

## Model Serving

Once you have a trained model object that you want to serve in production, you can deploy a
prediction endpoint using FastAPI or AWS Lambda functions:

- {ref}`Serving with FastAPI <serving_fastapi>`: Stand up a prediction microservice with FastAPI.
- {ref}`Serving with AWS Lambda <serving_aws_lambda>`: Create a serverless endpoint with AWS Lambda.
