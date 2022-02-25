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

(local_app)=

# Local Training and Prediction

In {ref}`Binding a Model and Dataset <model>` together, we defined a `Model` and `Dataset` object,
bound them together, and defined the core functions needed for model training and prediction.

In this guide, we'll learn how to interact with these objects locally to ensure that our code
is working as expected.

```{note}
Local interaction with `Model` objects are mainly useful for local development, debugging, and
unit testing of your `flytekit-learn` app.
```

Here's our complete `flytekit-learn` app for digit classification in a `main.py` script:

```{code-cell}
from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from flytekit_learn import Dataset, Model


dataset = Dataset(
    name="digits_dataset",
    test_size=0.2,
    shuffle=True,
    random_state=42,
    targets=["target"],
)

model = Model(
    name="digits_classifier",
    init=LogisticRegression,
    dataset=dataset
)

@dataset.reader
def reader(sample_frac: float = 1.0, random_state: int = 12345) -> pd.DataFrame:
    data = load_digits(as_frame=True).frame
    return data.sample(frac=sample_frac, random_state=random_state)


@model.trainer
def trainer(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())


@model.predictor
def predictor(estimator: LogisticRegression, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in estimator.predict(features)]


@model.evaluator
def evaluator(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return accuracy_score(target.squeeze(), predictor(estimator, features))
```

## Execute as a Python Module

We can then invoke the `model.train` method to train the sklearn estimator and `model.predict`
to generate predictions. Then invoke the app script with `python main.py`:

```{code-cell}
:tags: [remove-cell]

import warnings

warnings.simplefilter("ignore")
```

```{code-cell}
if __name__ == "__main__":
    trained_model, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        sample_frac=1.0,
        random_state=12345,
    )

    predictions = model.predict(
        trained_model,
        features=load_digits(as_frame=True).frame.sample(5, random_state=42)
    )

    print(f"trained model: {trained_model}")
    print(f"training metrics: {metrics}")
    print(f"predictions: {predictions}")
```

```{note}
You may notice a few things about the code example above:

- The `model.train` method takes the `dataset.reader` arguments as keyword-only arguments. In
  this case, the `sample_frac` and `random_state` values are passed into `model.train` and
  forwarded to `dataset.reader`.
- `model.train` returns a model instance and a dictionary of metrics.
  - The model instance is the same type as the return annotation of the `model.trainer` function,
    which in this case is `LogisticRegression`.
  - The metrics dictionary maps dataset split keys `{"train", "test"}` to metrics of the same
    type as the return annotation of the `model.evaluator`, which in this case is a `float`.
- The `model.predict` method accepts a `features` keyword argument containing the features
  of the same type defined in the `model.predictor` function.
```

## Serve with FastAPI

`flytekit-learn` integrates with [FastAPI](https://fastapi.tiangolo.com/) to make model serving super easy. Simply
create a `FastAPI` app and pass it into `model.serve` in the `main.py` script:

```{code-cell}
from fastapi import FastAPI

# dataset and model definition
...

app = FastAPI()

model.serve(app)
```

`model.serve` will take the `FastAPI` app and automatically create `/train/` and `/predict/` endpoints that you can
invoke with HTTP requests.

Start the server with `uvicorn`:

```{prompt} bash
:prompts: $

uvicorn main:app --reload
```

Once the server's started, you can use the Python `requests` library or any other HTTP library for training
and prediction:

```{code-block} python
import requests

# train a model
requests.post(
    "http://127.0.0.1:8000/train?local=True",
    json={
        "hyperparameters": {"C": 1.0, "max_iter": 1000},
        "sample_frac": 1.0,
        "random_state": 123,
    },
)

# generate predictions
requests.get(
    "http://127.0.0.1:8000/predict?local=True",
    json={"features": load_digits(as_frame=True).frame.sample(5, random_state=42).to_dict(orient="records")},
)
```

```{warning}
The `local=True` query parameter in the `/train` and `/predict` endpoint invocations
means that computation is being done on the app server itself, which, in this case,
is probably your laptop 💻.

This is fine for debugging, but it probably won't do in production, where the memory and
compute resource requirements are likely to exceed the resources available on the app
server.
```

## Next

We've run our training and prediction code by invoking our `flytekit-learn` app as a
python module and starting a local FastAPI server, but how do we deploy it as a suite of integrated
machine learning services in the ☁️ cloud?

`flytekit-learn` is coupled with [Flyte](https://docs.flyte.org/en/latest/), which is a scalable,
reliable, and robust orchestration platform for data processing and machine learning. But before we
deploy to the cloud, it's important to understand what a Flyte cluster is by
{ref}`spinning up a Flyte Sandbox <flyte_sandbox>` locally.
