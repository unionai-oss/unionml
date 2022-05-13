(serving_fastapi)=

# Serving with FastAPI

In the {ref}`Local Training and Prediction <local_serving_fastapi>` guide, we saw how to
create a prediction server locally using a model that we trained locally. This is great
for certain use cases, but it won't scale to bigger data or models.

In this guide, we'll learn how to create an online prediction server using a model
trained on a Flyte cluster.

## Prerequisites

Follow the {ref}`Flyte cluster <flyte_cluster>` guide to:

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

And that's it 🙌!

The `model_version` argument is `"latest"` by default, but you can serve other
models by passing in the unique identifier of the Flyte execution that produced a specific
model that you want to serve.

To list these prior model versions, do:

```{code-block} python
model.remote_list_model_versions(limit=5)
```

Or you can use the `unionml` cli:

```{prompt} bash
:prompts: $

unionml list-model-versions app:model --limit 5
```
