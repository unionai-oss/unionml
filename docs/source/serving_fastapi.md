(serving_fastapi)=

# Serving with FastAPI

In the {ref}`Local Training and Prediction <local_serving_fastapi>` guide, we saw how to
create a prediction server locally using a model that we trained locally. This is great
for certain use cases, but it won't scale to bigger data or models.

In this guide, we'll learn how to create an online prediction server using a model
trained on a Flyte cluster.

## Prerequisites

Follow the {ref}`Deploying to Flyte <flyte_cluster>` guide to:

1. Set up a local Flyte demo cluster.
2. Deploy a UnionML app on it.
3. Train a model on it.

## Serving a Model from a Flyte Backend

Once we've trained a model, the Flyte backend effectively becomes a model registry that
we can use to serve models, and the way we can do this is very similar to creating the
prediction service from a model that we've trained locally:

```{code-block} python
from fastapi import FastAPI

# dataset and model definition
...

app = FastAPI()

model.serve(app, remote=True, model_version="latest")
```

```{important}

   The `model_version` argument is `"latest"` by default, but you can serve other
   models by passing in the unique identifier of the Flyte execution that produced a specific
   model that you want to serve.

   To list these prior model versions, do:

   ```{code-block} python
   model.remote_list_model_versions(limit=5)
   ```

   Or you can use the UnionML cli:

   ```{prompt} bash
   :prompts: $

   unionml list-model-versions app:model --limit 5
   ```

```

Then start the server:

```{prompt} bash
:prompts: $

unionml serve app:app --reload
```

Once the server's started, you can use the Python `requests` library or any other HTTP library
to get predictions from input features:

```{code-block} python
import requests
from sklearn.datasets import load_digits

# generate predictions
requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": load_digits(as_frame=True).frame.sample(5, random_state=42).to_dict(orient="records")},
)
```

And that's it 🙌

## Next

Serving online predictions on FastAPI gives you full control over the server infrastructure, but
what if you want to standup a servless online prediction service? We'll see how we can achieve this
in the {ref}`Serving with AWS Lambda <serving_aws_lambda>` guide.
