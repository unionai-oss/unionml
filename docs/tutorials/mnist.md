---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# MNIST: Digits Classification

+++ {"tags": ["add-colab-badge"]}

...

+++

The MNIST dataset is considered to be the "hello world" dataset of machine
learning. It is a dataset of 60,000 small square 28×28 pixel grayscale images
of handwritten single digits between 0 and 9.

In that same spirit, we'll be making the "hello world" UnionML app using this
dataset and a simple linear classifier with [sklearn](https://scikit-learn.org/stable/index.html).

With this dataset, we'll see just how easy it is to create a single-script UnionML app.

```{note}
This tutorial is adapted from this [sklearn guide](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html).
```

```{code-cell}
:tags: [remove-cell]

%%capture
!pip install 'gradio<=3.0.10' pandas sklearn unionml
```

+++ {"tags": ["remove-cell"]}
> If you're running this notebook in google colab, you need to restart the
> kernel to make sure that the newly installed packages are correctly imported
> in the next line below.
+++

## Setup and importing libraries

First let's import our dependencies and create the UnionML `Dataset` and `Model` objects:

```{code-cell} ipython3
from typing import List, Union

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

dataset = Dataset(name="mnist_dataset", test_size=0.2, shuffle=True, targets=["class"])
model = Model(name="mnist_classifier", dataset=dataset)
```

Let's break down the code cell above.

We first define a {class}`~unionml.dataset.Dataset>`, which defines the specification for data that
can be used for training and prediction. We also give it a few keyword options:
- `test_size`: this indicated the percentage of the data that should be held
  over for testing. In this case the dataset is divided into test-set (20%) and
  training set (80%) for evaluation.
- `shuffle`: this randomly shuffles the data before splitting into train/test splits.
- `targets`: this accepts a list of strings referring to the column names of the dataset.

Then we define a {class}`~unionml.model.Model>`, which refers to the specification
for how to actually train the model, evaluate it, and generate predictions from
it. Note that we bind the `dataset` we just defined to the `model`.

## Caching Data

For convenience, we cache the dataset so that MNIST loading is faster upon
subsequent calls to the `fetch_openml` function:

```{code-cell} ipython3
from pathlib import Path
from joblib import Memory

memory = Memory(Path.home() / "tmp")
fetch_openml_cached = memory.cache(fetch_openml)
```

We do this so we don't have to re-download the dataset it every time we need to
train a model.

## Define Core UnionML Functions

Run the following command to define our core UnionML app functions:

```{code-cell} ipython3
@dataset.reader(cache=True, cache_version="1")
def reader() -> pd.DataFrame:
    dataset = fetch_openml_cached(
        "mnist_784",
        version=1,
        cache=True,
        as_frame=True,
    )
    # randomly sample a subset for faster training
    return dataset.frame.sample(1000, random_state=42)


@model.init
def init(hyperparameters: dict) -> Pipeline:
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )
    return estimator.set_params(**hyperparameters)


@model.trainer(cache=True, cache_version="1")
def trainer(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> Pipeline:
    return estimator.fit(features, target.squeeze())


@model.predictor
def predictor(
    estimator: Pipeline,
    features: pd.DataFrame,
) -> List[float]:
    return [float(x) for x in estimator.predict(features)]


@model.evaluator
def evaluator(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> float:
    return float(accuracy_score(target.squeeze(), estimator.predict(features)))
```

The `Dataset` and `Model` objects expose function decorators where we define
the behavior of our machine learning app:

- {meth}`~unionml.dataset.Dataset.reader` - Register a function for getting data
  from some external source.
- {meth}`~unionml.model.Model.init` - Register a function for initializing a
  model object. This is equivalent to specifying a class or callable using the
  `init` kwarg in the `Model` constructor.
- {meth}`~unionml.model.Model.trainer` - Register a function for training a
  model object.
- {meth}`~unionml.model.Model.predictor` - Register a function that generates
  predictions from a model object.
- {meth}`~unionml.model.Model.evaluator` - Register a function for evaluating given model object.

## Training a Model Locally

Then we can train our model locally:

```{code-cell} ipython3
estimator, metrics = model.train(
    hyperparameters={
        "classifier__penalty": "l2",
        "classifier__C": 0.1,
        "classifier__max_iter": 1000,
    }
)
print(estimator, metrics, sep="\n")
```

Note that we pass a dictionary of `hyperparameters` when we invoke
{meth}`evaluating <~unionml.model.Model.train>`,
which, in this case, follows the sklearn conventions for specifying
hyperparameters for [sklearn `Pipeline`s](https://scikit-learn.org/stable/modules/compose.html#nested-parameters)

## Serving on a Gradio Widget

Finally, let's create a `gradio` widget by simply using the
{meth}`~unionml.model.Model.predict` method in the `gradio.Interface`
object.

Before we do this, however, we want to define a {meth}`~unionml.dataset.Dataset.feature_loader`
function to handle the raw input coming from the `gradio` widget:

```{code-cell} ipython3
import numpy as np

@dataset.feature_loader
def feature_loader(data: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame(data.ravel())
        .transpose()
        .rename(columns=lambda x: f"pixel{x + 1}")
        .astype(float)
    )
```

We also need to take care to handle the `None` case when we press the `clear`
button on the widget using a `lambda` function:

```{code-cell} ipython3
:tags: [remove-output]

import gradio as gr

gr.Interface(
    fn=lambda img: img if img is None else model.predict(img)[0],
    inputs="sketchpad",
    outputs="label",
    live=True,
    allow_flagging="never",
).launch()
```

You might notice that the model may not perform as well as you might expect...
welcome to the world of machine learning practice! To obtain a better model
given a fixed dataset, feel free to play around with the model hyperparameters
or even switch up the model type/architecture that's defined in the `trainer`
function.
