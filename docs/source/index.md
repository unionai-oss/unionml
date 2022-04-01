<h1 style="font-weight: bold; font-size: 3.5em;">
    flytekit-<span style="color: var(--color-link)">learn</span>
</h1>

<div style="font-size: 1.5em; color: #777;">
The easiest way to build machine learning services.
</div>

<br>

```{toctree}
---
hidden: true
---
User Guide <user_guide>
Deployment Guide <deploying>
Contributing <contributing>
```

`flytekit-learn` is a framework for building end-to-end machine learning services.

It's built on top of [flyte](https://docs.flyte.org/en/latest/) and
[flytekit](https://docs.flyte.org/projects/flytekit/en/latest/) to provide a high-level
interface for model training and batch, streaming, and event-based prediction.

# Installation

```{code-block} bash
pip install flytekit-learn
```

# Getting Started

## Create a `flytekit-learn` App

A `flytekit-learn` app requires two core components: a `Dataset` and a `Model`.

First let's import our app dependencies and define `dataset` and `model` objects.
In this example, we'll build an app that classifies images of handwritten digits
into their corresponding digit labels.

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn/quickstart.py
      ---
      lines: 1-12
      ---
      ```

   ```

   ```{group-tab} pytorch

      ```{literalinclude} ../../tests/integration/sklearn/quickstart.py
      ---
      lines: 1-12
      ---
      ```

   ```

````

## Define App Methods

The `dataset` and `model` objects expose decorators that specify the
core components for model training and prediction:

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn/quickstart.py
      ---
      lines: 14-32
      ---
      ```

   ```

   ```{group-tab} pytorch

      ```{literalinclude} ../../tests/integration/sklearn/quickstart.py
      ---
      lines: 14-32
      ---
      ```

   ```

````

## Train and Predict Locally

Invoke `model.train` to train a model and `model.predict` to generate predictions.

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn/quickstart.py
      ---
      lines: 34-40
      ---
      ```

   ```

   ```{group-tab} pytorch

      ```{literalinclude} ../../tests/integration/sklearn/quickstart.py
      ---
      lines: 34-40
      ---
      ```

   ```

````

## Serve Seamlessly with FastAPI

`flytekit-learn` integrates with [FastAPI](https://fastapi.tiangolo.com/) to automatically
create `/train/` and `/predict/` endpoints. Start a server with `uvicorn` and call the app
endpoints with the `requests` library.

````{tabs}

   ```{group-tab} sklearn

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/sklearn/fastapi_app.py
      ---
      lines: 3-6
      ---
      ```

      Start the server, assuming the `flytekit-learn` app is in a `main.py` script

      ```{code-block} bash
      fklearn serve main:app --reload --model-path /tmp/model_object.joblib
      ```

      Invoke the endpoints using the `requests` library

      ```{literalinclude} ../../tests/integration/api_requests.py
      ```

   ```

   ```{group-tab} pytorch

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/sklearn/fastapi_app.py
      ---
      lines: 3-6
      ---
      ```

      Start the server, assuming the `flytekit-learn` app is in a `main.py` script

      ```{code-block} bash
      fklearn serve main:app --reload --model-path /tmp/model_object.joblib
      ```

      Invoke the endpoints using the `requests` library

      ```{literalinclude} ../../tests/integration/api_requests.py
      ```

   ```

````

# What Next?

Learn how to leverage the full power of `flytekit-learn` 🦾 in the {ref}`User Guide <user_guide>`
