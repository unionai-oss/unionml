---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(component_callbacks)=

# Component Callbacks

Component callbacks are user-defined functions that adhere to specific signatures depending
on the component.

## Prediction Callbacks

Prediction callbacks are defined as functions that take three arguments:

- The model object, which matches output type of the {meth}`unionml.model.Model.trainer` component.
- A batch of features, which matches {meth}`unionml.dataset.Dataset.feature_type`.
- A batch of predictions, which matches the output type of the {meth}`unionml.model.Model.predictor` component.

These callbacks are invoked in the {meth}`unionml.model.Model.predict` method after predictions have been
generated for a particular batch of features.

```{note}
Currently, callbacks apply to all contexts in which predictions occur including: batch prediction via Flyte,
online predictions in the FastAPI app or BentoML, and serverless predictions via AWS Lambda or BentoML.
```

```{code-cell} python
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from unionml import Model, Dataset

dataset = Dataset()
model = Model(dataset=dataset, init=LogisticRegression)


@dataset.reader
def reader() -> pd.DataFrame: ...


@model.trainer
def trainer(
    model_object: LogisticRegression,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> LogisticRegression:
    ...


def prediction_callback(
    model_object: LogisticRegression,
    features: pd.DataFrame,
    predictions: List[float],
):
    ...  # do something here


@model.predictor(callbacks=[prediction_callback])
def predictor(model_object: LogisticRegression, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in model_object.predict(features)]
```

Callbacks can be used for any purpose, but the primary use case is to log predictions using some third-party
package or service for the purposes of model monitoring.
