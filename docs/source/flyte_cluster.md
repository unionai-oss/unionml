(flyte_cluster)=

# Flyte Cluster

In {ref}`Local Training and Prediction <local_app>` we ran our training and prediction
functions locally by:

1. Executing our `unionml` app as python module by calling the `model.train` and
   `model.predict` methods.
2. Starting a FastAPI server and invoking `/predict` endpoint using
   the `requests` library.

```{note}
In the previous guides, we used our individual workstation to do the heavy lifting of
model training and prediction. While this might work well for prototyping, small datasets,
and light-weight models, this won't scale well to larger datasets and models. For that,
we'll need to access cloud resources.
```

`unionml` integrates tightly with [Flyte](https://docs.flyte.org/en/latest/), which is
a data- and machine-learning-aware orchestration platform that leverages cloud services like
[AWS](https://aws.amazon.com/) and [GCP](https://cloud.google.com/) to easily scale and
maintain data processing machine learning workloads.

In this guide, we'll:

1. Spin up a demo Flyte cluster, which is a standalone, minimal Flyte cluster that you can
   create on your individual laptop or workstation.
2. Configure digit classification `unionml` app to use the Flyte sandbox as the compute
   backend for our training and prediction workload.

## Prerequisites

First, install [`flytectl`](https://docs.flyte.org/projects/flytectl/en/latest/index.html#installation),
the command-line interface for Flyte, and [Docker](https://docs.docker.com/get-docker/), making sure
you have the Docker daemon running.

## Deploy App Workflows

A `unionml` app is composed of a `Dataset`, `Model`, and serving app component
(e.g. `fastapi.FastAPI`). Under the hood, a `Model` object exposes
`*_workflow` methods that return [`flytekit.workflow`](https://docs.flyte.org/projects/flytekit/en/latest/generated/flytekit.workflow.html#flytekit.workflow) objects, which are essentially
execution graphs that perform multiple steps of computation.

To make these computations scalable, reproducible, and auditable, we can serialize
our workflows and register them to a Flyte cluster, in this case a local Flyte demo cluster.

### Initializing a Digits Classification App

Going back to our digit classification app, let's assume that we've
{ref}`initialized our app <initialize>` using the `unionml init my_app` command
and have an `app.py` script with our digits classification model.

````{dropdown} See app.py

   ```{literalinclude} ../../unionml/templates/basic/{{cookiecutter.app_name}}/app.py
   ```

````

### Start a Local Flyte Demo Cluster

To start a Flyte demo cluster, run the following in your app directory:

```{prompt} bash
---
prompts: $
---

flytectl demo start --source .
```

```{note}
The `--source .` flag will initialize the Flyte demo cluster in a docker container with your app files
mounted inside. This is so that your app's workflows can be serialized and registered directly in the
Flyte sandbox.
```

We should now be able to go to `http://localhost:30080/console` on your browser to see the Flyte UI.


### The App Dockerfile

UnionML relies on [Docker](https://www.docker.com/) to package up all of your app's
source code and dependencies. The basic app template comes with a `Dockerfile`, which
we can use to do this:

````{dropdown} See Dockerfile

   ```{literalinclude} ../../unionml/templates/basic/{{cookiecutter.app_name}}/Dockerfile
   ---
   language: docker
   ---
   ```

````

### Configuring the Remote Backend

All you need to do to get your `unionml` app ready for deployment is to configure it with:

1. The Docker registry and image name that you want to use to package your app
2. The Flyte project and domain you want to use hosting your app's microservices.

In the `app.py` script, you can see the following code that does just this:

```{literalinclude} ../../tests/integration/sklearn_app/remote_config.py
---
lines: 5-10
---
```

```{important}
We've set the `config_file` argument to
`Path.home() / ".flyte" / "config.yaml"`, which was created automatically when
we invoked `flytectl demo start`.

Under the hood, `unionml` will handle the Docker build process locally, bypassing the
need to push your app image to a remote registry.
```

### Managing your Own Flyte Cluster

In this guide we're using the Flyte demo cluster that you can spin up on your local machine.
However, if you want to access to full power of Flyte, scaling to larger compute clusters
and accelerators for more data- and compute-heavy models, you can follow the
[Flyte Deployment Guides](https://docs.flyte.org/en/latest/deployment/index.html).

```{important}
To point your UnionML app to your own Flyte cluster, specify a `config.yaml` file in the `config_file`
argument that is properly configured to access that Flyte cluster. In this case, you'll also need to
specify a Docker registry that you have push access to via the `model.remote(registry="...")` keyword
argument.
```

## UnionML CLI

The `unionml` python package ships with the `unionml` cli, which we use to deploy the model and
invoke the training/prediction microservices that are automatically compiled by the `Dataset` and
`Model` objects.

### `unionml deploy`

To deploy, run:

```{prompt} bash
---
prompts: $
---

unionml deploy app:model
```

```{note}
The first argument of `unionml deploy` should be a `:`-separated string whose first section
is the module name containing the `unionml` app and second section is the variable
name pointing to the `unionml.Model` object.
```

Now that your app workflows are deployed, you can run training and prediction jobs using
the Flyte sandbox cluster:

### `unionml train`

Train a model given some hyperparameters:

```{prompt} bash
---
prompts: $
---

unionml train app:model -i '{"hyperparameters": {"C": 1.0, "max_iter": 1000}}'
```

### `unionml predict`

Generate predictions with json data:

```{prompt} bash
---
prompts: $
---

unionml predict app:model -f <path-to-json-file>
```

Where `<path-to-json-file>` is a json file containing feature data that's compatible with the model.

```{note}
Currently, only json records data that can be converted to a pandas DataFrame is supported.
```


## Programmatic API

UnionML also provides a programmatic API to deploy your app and kick off training and prediction jobs
that run on the Flyte cluster. We simply import the UnionML `Model` object into another python module:

### `model.remote_deploy`

```{code-block} python
from app import model

model.remote_deploy()
```

### `model.remote_train`

```{code-block} python
from unionml.model import ModelArtifact

from app import model

model_artifact: ModelArtifact = model.remote_train(
    hyperparameters={"C": 1.0, "max_iter": 1000},
)
```

The `model_artifact` output `NamedTuple` contains three attributes:
- `model_object`: The actual trained model object, in this case an sklearn `BaseEstimator`.
- `hyperparameters`: The hyperparameters used to train the model.
- `metrics`: The metrics associated with the training and test set of the dataset used during training.

```{note}
By default, invoking `remote_train` is a blocking operation, i.e. the python process will wait until
the Flyte backend completes training.
```

### `model.remote_predict`

```{code-block} python
from sklearn.datasets import load_digits

from app import model

features = load_digits(as_frame=True).frame.sample(5, random_state=42)
predictions = model.remote_predict(features=features)
```

```{note}
The `features` kwarg should be the same type as the output type of the `dataset.reader`
function. In this case, that would be a `pandas.DataFrame`.
```
