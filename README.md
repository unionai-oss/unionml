<p align="center">
  <img src="https://raw.githubusercontent.com/unionai-oss/unionml/main/docs/source/_static/images/union-logo.svg" alt="Union.ai Logo" width="100">
</p>

<h1 align="center">UnionML</h1>

<p align="center">
    <strong>The easiest way to build and deploy machine learning microservices</strong>
</p>

---

<br>

[![PyPI version shields.io](https://img.shields.io/pypi/v/unionml.svg?color=blue)](https://pypi.org/project/unionml/)
[![Documentation Status](https://readthedocs.org/projects/unionml/badge/?version=latest)](https://unionml.readthedocs.io/en/latest/?badge=latest)
[![Python application](https://github.com/unionai-oss/unionml/actions/workflows/build.yml/badge.svg)](https://github.com/unionai-oss/unionml/actions/workflows/build.yml)

<br>


**UnionML** is an open source MLOps framework that aims to reduce the boilerplate and friction
that comes with building models and deploying them to production.

You can create **UnionML Apps** by defining a few core methods that are automatically bundled
into ML microservices, starting with model training and offline and online prediction.

Built on top of [Flyte](https://docs.flyte.org/en/latest/), UnionML provides a high-level
interface for productionizing your ML models so that you can focus on curating a better dataset
and improving your models.

To learn more, check out the 📖 [documentation](https://unionml.readthedocs.io).

## Installing

Install using pip:

```bash
pip install unionml
```

## A Simple Example

Create a `Dataset` and `Model`, which together form a **UnionML App**:

```python
from unionml import Dataset, Model

from sklearn.linear_model import LogisticRegression

dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=LogisticRegression, dataset=dataset)
```

Define `Dataset` and `Model` methods for training a hand-written digits classifier:

```python
from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

@dataset.reader
def reader() -> pd.DataFrame:
    return load_digits(as_frame=True).frame

@model.trainer
def trainer(
    estimator: LogisticRegression,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())

@model.predictor
def predictor(
    estimator: LogisticRegression,
    features: pd.DataFrame
) -> List[float]:
    return [float(x) for x in estimator.predict(features)]

@model.evaluator
def evaluator(
    estimator: LogisticRegression,
    features: pd.DataFrame,
    target: pd.DataFrame
) -> float:
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))
```

And that's all ⭐️!

By defining these four methods, you've created a minimal **UnionML App** that you can:

- [Execute locally](https://unionml.readthedocs.io/en/latest/index.html#train-and-predict-locally) to debug and iterate on your code.
- [Serve Seamlessly with FastAPI](https://unionml.readthedocs.io/en/latest/index.html#serve-seamlessly-with-fastapi) for online prediction.
- [Deploy on a Flyte Cluster](https://unionml.readthedocs.io/en/latest/deploying.html) to scale your model training and schedule offline prediction jobs.

## Contributing

All contributions are welcome 🤝 ! Check out the [contribution guide](https://unionml.readthedocs.io/en/latest/contributing.html) to learn more about how to contribute.
