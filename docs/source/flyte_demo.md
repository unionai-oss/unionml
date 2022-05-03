(flyte_demo)=

# Flyte Demo Cluster

In {ref}`Local Training and Prediction <local_app>` we ran our training and prediction
functions locally by:

1. Executing our `unionml` app as python module by calling the `model.train` and
   `model.predict` methods.
2. Starting a FastAPI server and invoking the `/train` and `/predict` endpoints using
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

First, install [Docker](https://docs.docker.com/get-docker/) and
[`flytectl`](https://docs.flyte.org/projects/flytectl/en/latest/index.html#installation),
the command-line interface for Flyte.


## Deploy App Workflows

A `unionml` app is composed of a `Dataset`, `Model`, and serving app component
(e.g. `fastapi.FastAPI`). Under the hood, a `Model` object exposes
`*_workflow` methods that return [`flytekit.workflow`](https://docs.flyte.org/projects/flytekit/en/latest/generated/flytekit.workflow.html#flytekit.workflow) objects, which are essentially
execution graphs that perform multiple steps of computation.

To make these computations scalable, reproducible, and auditable, we can serialize
our workflows and register them to a Flyte cluster, in this case a local Flyte sandbox.

Going back to our digit classification app, let's assume that we have a `unionml`
app in an `app.py` file. We can use the `unionml` command-line tool to easily deploy
our app workflows.

````{dropdown} See app source

   ```{literalinclude} ../../tests/integration/sklearn_app/quickstart.py
   ```

````
### Creating a Dockerfile

UnionML relies on [Docker](https://www.docker.com/) to package up all of your app's
source code and dependencies. Create a `Dockerfile` for your app with the following
contents:

```{code-block} docker
FROM python:3.8-slim-buster

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get update && apt-get install -y build-essential git-all

# Install the AWS cli separately to prevent issues with boto being written over
RUN pip3 install awscli

RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
COPY ./requirements.txt /root
RUN pip install -r /root/requirements.txt

# Copy the actual code
COPY . /root
```

### Configuring the Remote Backend

All you need to get your `unionml` app for deployment is to configure it with the
Docker registry and image name that you want to use to package you app, as well as the
Flyte project and domain you want to use hosting your app's microservices.

Add the following code anywhere in your app script:

```{literalinclude} ../../tests/integration/sklearn_app/remote_config.py
---
lines: 3-10
---
```

```{important}
We're set the `config_file_path` argument to
`Path.home() / ".flyte" / "config-sandbox.yaml"`, which was created automatically when
we invoked `flytectl demo start`.

Under the hood, `unionml` will handle the Docker build process locally, bypassing the
need to push your app image to a remote registry.
```

## Initialize a Flyte Sandbox

Then in your app directory, run:

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
