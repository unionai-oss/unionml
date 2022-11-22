(deploying_to_production)=

# Deployment Guides

```{toctree}
---
hidden: true
---
flyte_cluster
serving_fastapi
serving_aws_lambda
serving_bentoml
reacting_to_s3_events
```

Currently, UnionML apps support two core types of machine learning microservices: model training and
model serving.

## Production Training and Batch Predictions

UnionML uses `flytekit` under the hood to execute your training and prediction workflows locally, but you
can benefit from the reproducibility and scalability benefits of UnionML by deploying your workflows
to a production-grade `flyte` cluster.

- {ref}`Deploy to a Flyte Cluster<flyte_cluster>`: Deploy training and prediction services to a Flyte cluster.

## Serving Online Predictions

Once you have a trained model object that you want to serve in production, you can:

- {ref}`Serve with FastAPI <serving_fastapi>`: Stand up an online prediction service with FastAPI.
- {ref}`Serve with AWS Lambda <serving_aws_lambda>`: Create an online prediction serverless endpoint with AWS Lambda.
- {ref}`Serve with BentoML <serving_bentoml>`: Leverage [BentoML](https://docs.bentoml.org/en/latest/) to deploy a prediction service to a wide variety of cloud platforms.

## Serving Reactive Predictions

Some predictive applications require reacting to events that occur in some external system:

- {ref}`Reacting to S3 Events <reacting_to_s3_events>`: Generate predictions in response to files being dumped into a specified S3 path.
