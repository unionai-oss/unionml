# Basic Example

This project contains source code for building and deploying a serverless application
that you can run locally and serve via FastAPI.

It includes the following files and folders:

- `app.py`: The UnionML app code for digits classification.
- `Dockerfile`: Dockerfile to package up the UnionML app for AWS Lambda serving.
- `requirements.txt`: Python dependencies of the app.
- `data`: Sample data used for invoking prediction endpoints.

The code in `app.py` implements a hand-written digits model, but you can adapt the code to
fit your use case.

To learn more about how a UnionML App is structured, check out the
📖 [Documentation](https://unionml.readthedocs.io/en/latest/basics.html).

```
curl -X POST \
    -H "Content-Type: application/json" \
    --data "$(cat sample_features.json)" \
    http://0.0.0.0:3000/predict
```

```
URL=$(terraform output -json | jq -r .endpoint.value)predict
curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data "$(cat data/sample_features.json)" \
  $URL
```
