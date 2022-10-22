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

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology (MNIST) dataset. It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The MNIST dataset is considered to be the "hello world" example of machine learning. In that same spirit, we'll be making the "hello world" UnionML app using this dataset and a simple linear classifier with [sklearn](https://scikit-learn.org/stable/index.html).

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
> If you're running this notebook in google colab, you need to restart the kernel to make sure that the newly installed packages are correctly imported in the next line below.
+++

## Step 1: Setup and importing libraries

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
model = Model(name="mnist_classifier", init=LogisticRegression, dataset=dataset)
```

Here are the following functions used above:

- `dataset` - It is the central source that consist of multiple rows and columns of data, to be used for training a model.
- `model` - It refers to the training of the machine to predict or recognize patterns. Here, Logistic Regression is initiated (`init`), which predicts the probability of a binary (yes/no) event occurring.
- `test-size` - It is the function that indicates the percentage of the data that should be held over for testing. In this case the dataset is divided into test-set (20%) and training set (80%) for evaluation.
- `shuffle` - It randomly shuffles data from a dataset within attributes (column of a dataset) before validation split.
- `targets` - It is an argument that accepts a list of strings referring to the column names.

## Step 2: Storage and caching of data

For convenience, we cache the dataset so that MNIST loading is faster upon subsequent calls to the `fetch_openml` function:

```{code-cell} ipython3
from pathlib import Path
from joblib import Memory

memory = Memory(Path.home() / "tmp")
fetch_openml_cached = memory.cache(fetch_openml)
```

Here we use `/ "temp"` for shortlife fast storage. The function `fetch_openml` caches the HTTP result from the OpenML server.

## Step 3: Setup of UnionML Core functions

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

The above functions help you create a UnionML App with features like:

- `dataset.reader()` - Register a [reader](https://unionml.readthedocs.io/en/latest/generated_api_reference/unionml.dataset.Dataset.html#unionml.dataset.Dataset.reader) function for getting data from some external source.
- `model.init` - Register a function for [initializing](https://unionml.readthedocs.io/en/latest/generated_api_reference/unionml.model.Model.html#unionml.model.Model.init) a model object.
- `model.trainer` - Register a function for [training](https://unionml.readthedocs.io/en/latest/generated_api_reference/unionml.model.Model.html#unionml.model.Model.trainer) a model object.
- `model.predictor` - Register a function that generates [predictions](https://unionml.readthedocs.io/en/latest/generated_api_reference/unionml.model.Model.html#unionml.model.Model.predictor) from a model object.
- `model.evaluator` - Register a function for [producing metrics](https://unionml.readthedocs.io/en/latest/generated_api_reference/unionml.model.Model.html#unionml.model.Model.evaluator) for given model object.

## Step 4: Training a Model Locally

Then we can train our model locally:

```{code-cell} ipython3
estimator, metrics = model.train(
    hyperparameters={
        "classifier__penalty": "l2",
        "classifier__C": 0.1,
        "classifier__max_iter": 1000,
    }
)
features = reader().sample(5, random_state=42).drop(["class"], axis="columns")
print(estimator, metrics, sep="\n")
```

The above codes represent the following functions and arguments:

- `hyperparameters` - Parameters that control the learning process of the model and defines the parameters which the model will ultimately learn.
- `classifier__penalty` - It imposes a penalty to the logistic model for having too many variables. Here, L2 regularization penalizes the sum of squares of the weights.
- `classifier__C` - It is inverse of regularization strength. Smaller values specify stronger regularization.
- `classifier__max_iter` - It determines the maximum number of passes over the training data (aka epochs). By default it is set to 1,000.
- `sample()` - It returns a random sample of items from an axis of object.
- `random_state` - Argument if set to a particular integer, will return same rows as sample in every iteration.
- `drop()` - Function returns a new object with labels in requested axis removed (`labels` : Index or column labels to drop; `axis` : Whether to drop labels from the index (0 / "index") or columns (1 / "columns")).

## Step 5: Serving on a Gradio Widget

Finally, let's create a `gradio` widget by simply using the [`model.predict`](https://unionml.readthedocs.io/en/latest/generated_api_reference/unionml.model.Model.html#unionml.model.Model.predict) method into the `gradio.Interface` object.

Before we do this, however, we want to define a [`feature_loader`](https://unionml.readthedocs.io/en/latest/dataset.html#feature-loader) function to handle the raw input coming from the `gradio` widget:

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

We also need to take care to handle the `None` case when we press the `clear` button on the widget using a `lambda` function:

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

You might notice that the model may not perform as well as you might expect...welcome to the world of machine learning practice! To obtain a better model given a fixed dataset, feel free to play around with the model hyperparameters or even switch up the model type/architecture that's defined in the `trainer` function.
