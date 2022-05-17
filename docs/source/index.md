<h1 style="font-weight: bold; font-size: 3.5em;">
   UnionML
</h1>

<div style="font-size: 1.5em; color: #777;">
   The easiest way to build and deploy machine learning microservices
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

**UnionML** is an open source MLOps framework that aims to reduce the boilerplate and friction
that comes with building models and deploying them to production.

You can create **UnionML Apps** by defining a few core methods that are automatically bundled
into ML microservices, starting with model training and offline and online prediction.

Built on top of [Flyte](https://docs.flyte.org/en/latest/), UnionML provides a high-level
interface for productionizing your ML models so that you can focus on curating a better dataset
and improving your models.

# Installation

```{code-block} bash
pip install unionml
```

# Quickstart

## A Minimal UnionML App

A `unionml` app requires two core components: a `Dataset` and a `Model`.

First let's import our app dependencies and define `dataset` and `model` objects.
In this example, we'll build an app that classifies images of handwritten digits
into their corresponding digit labels.

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn_app/quickstart.py
      ---
      lines: 1-12
      ---
      ```

   ```

   ```{group-tab} pytorch

      ```{literalinclude} ../../tests/integration/pytorch_app/quickstart.py
      ---
      lines: 1-28
      ---
      ```

   ```

   ```{group-tab} keras

      ```{literalinclude} ../../tests/integration/keras_app/quickstart.py
      ---
      lines: 1-25
      ---
      ```

   ```

````

## Define App Methods

The `dataset` and `model` objects expose decorators that specify the
core components for model training and prediction:

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn_app/quickstart.py
      ---
      lines: 14-32
      ---
      ```

   ```

   ```{group-tab} pytorch

      First we'll define some helper functions to convert dataframes to tensors

      ```{literalinclude} ../../tests/integration/pytorch_app/quickstart.py
      ---
      lines: 31-37
      ---
      ```

      Then let's define the app methods

      ```{literalinclude} ../../tests/integration/pytorch_app/quickstart.py
      ---
      lines: 39-75
      ---
      ```

   ```

   ```{group-tab} keras

      ```{literalinclude} ../../tests/integration/keras_app/quickstart.py
      ---
      lines: 28-66
      ---
      ```

   ```

````

## Train and Predict Locally

Invoke `model.train` to train a model and `model.predict` to generate predictions.

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn_app/quickstart.py
      ---
      lines: 34-40
      ---
      ```

   ```

   ```{group-tab} pytorch

      ```{literalinclude} ../../tests/integration/pytorch_app/quickstart.py
      ---
      lines: 78-87
      ---
      ```

   ```

   ```{group-tab} keras

      ```{literalinclude} ../../tests/integration/keras_app/quickstart.py
      ---
      lines: 69-78
      ---
      ```

   ```

````

## Serve Seamlessly with FastAPI

`unionml` integrates with [FastAPI](https://fastapi.tiangolo.com/) to automatically
create `/train/` and `/predict/` endpoints. Start a server with `unionml serve` and call the app
endpoints with the `requests` library.

````{tabs}

   ```{group-tab} sklearn

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/sklearn_app/fastapi_app.py
      ---
      lines: 1,4-6
      ---
      ```

      Start the server, assuming the `unionml` app is in a `main.py` script

      ```{code-block} bash
      unionml serve main:app --reload --model-path /tmp/model_object.joblib
      ```

   ```

   ```{group-tab} pytorch

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/pytorch_app/fastapi_app.py
      ---
      lines: 1,4-6
      ---
      ```

      Start the server, assuming the `unionml` app is in a `main.py` script

      ```{code-block} bash
      unionml serve main:app --reload --model-path /tmp/model_object.pt
      ```

   ```

   ```{group-tab} keras

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/keras_app/fastapi_app.py
      ---
      lines: 1,4-6
      ---
      ```

      Start the server, assuming the `unionml` app is in a `main.py` script

      ```{code-block} bash
      unionml serve main:app --reload --model-path /tmp/model_object.h5
      ```

   ```

````

Then you can invoke the endpoints using the `requests` library

```{literalinclude} ../../tests/integration/api_requests.py
```

# What Next?

Learn how to leverage the full power of `unionml` 🦾 in the {ref}`User Guide <user_guide>`


```{admonition} Want to contribute?
Check out the {ref}`Contributing Guide <contributing>`.
```
