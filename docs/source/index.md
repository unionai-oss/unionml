<h1 style="font-weight: bold; font-size: 3.5em;">
   UnionML
</h1>

<div style="font-size: 1.5em; color: #777;">
   The easiest way to build and deploy machine learning microservices
</div>

<br>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unionml?style=for-the-badge)](https://pypi.org/project/unionml/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/unionml?style=for-the-badge)](https://pypi.org/project/unionml/)
[![Documentation Status](https://img.shields.io/readthedocs/unionml/latest?style=for-the-badge)](https://unionml.readthedocs.io/en/latest/?badge=latest)
[![Build](https://img.shields.io/github/actions/workflow/status/unionai-oss/unionml/build.yml?branch=main&style=for-the-badge)](https://github.com/unionai-oss/unionml/actions/workflows/build.yml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/unionml?style=for-the-badge)](https://pypistats.org/packages/unionml)
[![Roadmap](https://img.shields.io/badge/Project-Roadmap-blueviolet?style=for-the-badge)](https://github.com/orgs/unionai-oss/projects/1/views/4)
[![OSS Planning](https://img.shields.io/badge/Event-OSS_Planning-yellow?style=for-the-badge)](https://app.addevent.com/event/tj14110550/)


```{toctree}
---
hidden: true
---
Basics <basics>
Deployment Guides <deploying>
Advanced Guides <advanced_guides>
```

```{toctree}
---
hidden: true
caption: Tutorials
---
Tutorials <tutorials_intro>
Computer Vision <computer_vision>
```

```{toctree}
---
hidden: true
caption: References
---
api_reference
cli_reference
```

```{toctree}
---
hidden: true
caption: Resources
---

Contributing Guide <contributing_guide>
Union.ai <https://www.union.ai/unionml>
```

<br>

**UnionML** is an open source MLOps framework that reduces the boilerplate, complexity,
and friction that comes with building models and deploying them to production. Taking inspiration from
web protocols, UnionML asks the question:

> Is it possible to define a standard set of functions/methods for machine learning that can be
> reused in many different contexts, from model training to prediction?

UnionML aims to unify the ever-evolving ecosystem of machine learning and data tools into a single
interface for expressing microservices as Python functions.

You can create _UnionML Apps_ by defining a few core methods that are automatically bundled into
_ML microservices_, starting with model training and offline/online prediction:

```{mermaid}
   :align: center

   %%{init: {'theme':'default'}}%%

   flowchart LR

      A[UnionML App]

      subgraph methods
      Rm[reader]
      Tm[trainer]
      Pm[predictor]
      Em[...]
      end

      subgraph microservices
      T[train]
      Pb[batch predict]
      Po[online predict]
      E[...]
      end

      Rm --> A
      Tm --> A
      Pm --> A
      Em --> A

      A --> T
      A --> Pb
      A --> Po
      A --> E
```

Brought to you by the [Union.ai](https://www.union.ai/) team, UnionML is built on top of
[Flyte](https://docs.flyte.org/en/latest/) to provide a high-level interface for productionizing
your ML models so that you can focus on curating a better dataset and improving your models.

# Installation

```{prompt} bash
:prompts: $

pip install unionml
```

````{admonition} For M1 Mac users
:class: important

Before installing `unionml`, follow these StackOverflow posts to install
[numpy](https://numpy.org/) and [pandas](https://pandas.pydata.org/pandas-docs/stable/):

- [numpy installation](https://stackoverflow.com/a/65581354)
- [pandas installation](https://stackoverflow.com/a/66048187)
````

# Quickstart

A UnionML app is composed of two core classes: a {class}`~unionml.dataset.Dataset` and a
{class}`~unionml.model.Model`.

In this example, we'll build a minimal UnionML app that classifies images of handwritten digits
into their corresponding digit labels using [sklearn](https://scikit-learn.org/stable/), [xgboost](https://xgboost.ai/)
[pytorch](https://pytorch.org/), or [keras](https://keras.io/).

Create a python file called `app.py`, import app dependencies, and define `dataset` and `model` objects.

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn_app/quickstart.py
      ---
      lines: 1-12
      ---
      ```

   ```
   ```{group-tab} xgboost

      Install [xgboost](https://xgboost.ai/):

      ```{prompt} bash
      :prompts: $

      pip install xgboost
      ```

      ```{literalinclude} ../../tests/integration/xgboost_app/quickstart.py
      ---
      lines: 1-24
      ---
      ```

   ```  
   
   ```{group-tab} pytorch

      Install [pytorch](https://pytorch.org/):

      ```{prompt} bash
      :prompts: $

      pip install torch
      ```

      ```{literalinclude} ../../tests/integration/pytorch_app/quickstart.py
      ---
      lines: 1-28
      ---
      ```

   ```

   ```{group-tab} keras

      Install [keras](https://keras.io/) via [tensorflow](https://www.tensorflow.org/):

      ```{prompt} bash
      :prompts: $

      pip install tensorflow
      ```

      ````{admonition} For M1 Mac users
      :class: important

      Follow [this StackOverflow post](https://stackoverflow.com/questions/66741778/how-to-install-h5py-needed-for-keras-on-macos-with-m1) to install tensorflow on an M1 Mac
      ````

      ```{literalinclude} ../../tests/integration/keras_app/quickstart.py
      ---
      lines: 1-25
      ---
      ```

   ```

````

## Define App Methods

Specify the core functions for training and prediction with the decorators
exposed by the `dataset` and `model` objects:

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn_app/quickstart.py
      ---
      lines: 14-32
      ---
      ```

   ```

   ```{group-tab} xgboost

      ```{literalinclude} ../../tests/integration/xgboost_app/quickstart.py
      ---
      lines: 27-44
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

Invoke {meth}`~unionml.model.Model.train` to train a model and {meth}`~unionml.model.Model.predict`
to generate predictions.

````{tabs}

   ```{group-tab} sklearn

      ```{literalinclude} ../../tests/integration/sklearn_app/quickstart.py
      ---
      lines: 34-40
      ---
      ```

   ```

   ```{group-tab} xgboost

      ```{literalinclude} ../../tests/integration/xgboost_app/quickstart.py
      ---
      lines: 46-52
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

UnionML integrates with [FastAPI](https://fastapi.tiangolo.com/) to automatically
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

      Start the server, assuming the UnionML app is in a `app.py` script

      ```{code-block} bash
      unionml serve app:app --reload --model-path /tmp/model_object.joblib
      ```

   ```

   ```{group-tab} xgboost

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/xgboost_app/fastapi_app.py
      ---
      lines: 1,4-6
      ---
      ```

      Start the server, assuming the UnionML app is in a `app.py` script

      ```{code-block} bash
      unionml serve app:app --reload --model-path /tmp/model_object.pickle
      ```

   ```

   ```{group-tab} pytorch

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/pytorch_app/fastapi_app.py
      ---
      lines: 1,4-6
      ---
      ```

      Start the server, assuming the UnionML app is in a `main.py` script

      ```{code-block} bash
      unionml serve app:app --reload --model-path /tmp/model_object.pt
      ```

   ```

   ```{group-tab} keras

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/keras_app/fastapi_app.py
      ---
      lines: 1,4-6
      ---
      ```

      Start the server, assuming the UnionML app is in a `main.py` script

      ```{code-block} bash
      unionml serve app:app --reload --model-path /tmp/model_object.h5
      ```

   ```

````

```{important}
The first argument to `unionml serve` is a `:`-separated string where the first
part is the module name of the app script, and the second part is the variable
name of the FastAPI app.
```

Then you can invoke the endpoints using the `requests` library, e.g. in a separate
`client.py` script:

```{literalinclude} ../../tests/integration/api_requests.py
```

# What Next?

Learn how to leverage the full power of UnionML 🦾 in the {ref}`Basics <basics>` guide.


```{admonition} Want to contribute?
:class: important

Check out the {ref}`Contributing Guide <contributing_guide>`.
```
