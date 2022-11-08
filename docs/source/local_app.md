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
unit testing of your UnionML app.
```

Here's our complete UnionML app for digit classification in a `app.py` script:

```{code-cell} python
from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model


dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=LogisticRegression, dataset=dataset)


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
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))
```

## Execute as a Python Module

We can then invoke the `model.train` method to train the sklearn estimator and `model.predict`
to generate predictions. Then invoke the app script with `python app.py`:

```{code-cell} python
:tags: [remove-cell]

import warnings

warnings.simplefilter("ignore")
```

```{code-cell}
if __name__ == "__main__":
    model_object, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 10000},
        sample_frac=1.0,
        random_state=12345,
    )

    predictions = model.predict(
        features=load_digits(as_frame=True).frame.sample(5, random_state=42)
    )

    print(f"model object: {model_object}")
    print(f"training metrics: {metrics}")
    print(f"predictions: {predictions}")

    # save model to a file, using joblib as the default serialization format
    model.save("/tmp/model_object.joblib")
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
- At the end of the file we save the model object to a file called `/tmp/model_object.joblib`. This is
  simply an `sklearn` base estimator that you know and love!
```

(local_serving_fastapi)=

## Serve with FastAPI

UnionML integrates with [FastAPI](https://fastapi.tiangolo.com/) to make model serving super easy. Simply
create a `FastAPI` app and pass it into `model.serve` in the `app.py` script:

```{code-cell} python
from fastapi import FastAPI

# dataset and model definition
...

app = FastAPI()

model.serve(app)
```

`model.serve` will take the `FastAPI` app and automatically create a `/predict/` endpoint that you can
invoke with HTTP requests.

Start the server with <a href="cli_reference.html#unionml-serve">unionml serve</a>

```{prompt} bash
:prompts: $

unionml serve app:app --model-path /tmp/model_object.joblib --reload
```

```{note}
The `--model-path` option points to a local file containing the serialized model object that
we created above when we executed the UnionML app script.
```

Once the server's started, you can use the Python `requests` library or any other HTTP library
to get predictions from input features. For example, you can copy the following code into a `client.py`
script to generate predictions from the endpoint:

```{code-block} python
import requests

# generate predictions
requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": load_digits(as_frame=True).frame.sample(5, random_state=42).to_dict(orient="records")},
)
```

```{note}
The `/predict` endpoint computation is being done on the app server itself, which, in this case, is probably
your laptop 💻. You'll need to ensure that your prediction server has the resources needed to load the model
into memory and generate predictions.
```

## Next

We've run our training and prediction code by invoking our UnionML app as a
python module and starting a local FastAPI server, but how do we deploy it as a suite of integrated
machine learning services in the ☁️ cloud?

UnionML is coupled with [Flyte](https://docs.flyte.org/en/latest/), which is a scalable,
reliable, and robust orchestration platform for data processing and machine learning. But before we
deploy to the cloud, it's important to understand what a Flyte cluster is by
{ref}`spinning up a Flyte Cluster <flyte_cluster>` locally.
