(flyte_sandbox)=

# Flyte Sandbox

In {ref}`Local Training and Prediction <local_app>` we ran our training and prediction
functions locally by:

1. Executing our `flytekit-learn` app as python module by calling the `model.train` and
   `model.predict` methods.
2. Starting a FastAPI server and invoking the `/train` and `/predict` endpoints using
   the `requests` library.

```{note}
In the previous guides, we used our individual workstation to do the heavy lifting of
model training and prediction. While this might work well for prototyping, small datasets,
and light-weight models, this won't scale well to larger datasets and models. For that,
we'll need to access cloud resources.
```

`flytekit-learn` integrates tightly with [Flyte](https://docs.flyte.org/en/latest/), which is
a data- and machine-learning-aware orchestration platform that leverages cloud services like
[AWS](https://aws.amazon.com/) and [GCP](https://cloud.google.com/) to easily scale and
maintain data processing machine learning workloads.

In this guide, we'll:

1. Spin up a Flyte sandbox, which is a standalone, minimal Flyte cluster that you can create
   on your individual laptop or workstation.
2. Configure digit classification `flytekit-learn` app to use the Flyte sandbox as the compute
   backend for our training and prediction workload.


## Initialize a Flyte Sandbox

Install [`flytectl`](https://docs.flyte.org/projects/flytectl/en/latest/index.html),
the commandline interface for Flyte.

````{tabs}

   ```{tab} Brew (OSX)

      ```{prompt} bash
      ---
      prompts: $
      ---

      brew install flyteorg/homebrew-tap/flytectl
      ```

   ```

   ```{tab} Curl

      ```{prompt} bash
      ---
      prompts: $
      ---

      curl -sL https://ctl.flyte.org/install | sudo bash -s -- -b /usr/local/bin # You can change path from /usr/local/bin to any file system path
export PATH=$(pwd)/bin:$PATH # Only required if user used different path then /usr/local/bin

      ```

   ```

````

Then in your app directory, run:

```{prompt} bash
---
prompts: $
---

flytectl sandbox start --source .
```

```{note}
The ``--source .`` flag will initialize the Flyte sandbox a Docker container
with the source files of your app mounted. This is so that your app's workflows
can be compiled and registered directly in the Flyte sandbox.
```

```{note}
Having trouble getting Flyte sandbox to start? See the [troubleshooting guide](https://docs.flyte.org/en/latest/community/troubleshoot.html#troubleshooting-guide)
```

We should now be able to go to `http://localhost:30081/console` on your browser to see the Flyte UI.


## Deploy App Workflows

A `flytekit-learn` app is composed of a `Dataset`, `Model`, and serving app component
(e.g. `fastapi.FastAPI`). Under the hood, a `Model` object exposes
`*_workflow` methods that return [`flytekit.workflow`](https://docs.flyte.org/projects/flytekit/en/latest/generated/flytekit.workflow.html#flytekit.workflow) objects, which are essentially
execution graphs that perform multiple steps of computation.

To make these computations scalable, reproducible, and auditable, we can serialize
our workflows and register them to a Flyte cluster, in this case a local Flyte sandbox.

Going back to our digit classification app, let's assume that we have a `flytekit-learn`
app in an `app/main.py` file. We can use the `fklearn` command-line tool to easily deploy
our app workflows.

````{dropdown} See app source

   ```{literalinclude} ../../tests/integration/sklearn/quickstart.py
   ```

````

### Configuration

Create a file called `flyte.config`, then add the following configuration settings:

```{code-block} ini
[sdk]
workflow_packages=app.main  # module name of the app

[auth]
raw_output_data_prefix=s3://my-s3-bucket/raw_output

[platform]
url=localhost:30081
insecure=true

[aws]
endpoint=http://localhost:30084
access_key_id=minio
secret_access_key=miniostorage
```

### Dockerfile

Flyte relies on [Docker](https://www.docker.com/) to package up all of your app's
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

# This tag is supplied by the fklearn deploy command
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
ENV FLYTE_CONFIG=config/remote.config
```

### `fklearn deploy`

To deploy run:

```{prompt} bash
---
prompts: $
---

fklearn deploy app.main:model -i "flytekit-learn:v0" -v 0  # deploy flytekit-learn model
```

```{note}
The first argument of `fklearn deploy` should be a `:`-separated string whose first section
is the module name containing the `flytekit-learn` app and second section is the variable
name pointing to the `flytekit_learn.Model` object.
```

Now that your app workflows are deployed, you can run training and prediction jobs using
the Flyte sandbox cluster:

### `fklearn train`

Train a model given some hyperparameters:

```{code-block} bash
fklearn train app.main:model -i '{"hyperparameters": {"C": 1.0, "max_iter": 1000}, "sample_frac": 1.0, "random_state": 123}'
```

### `fklearn predict`

Generate predictions with json data:

```{code-block} bash
fklearn predict app.main:model -f <path-to-json-file>
```
