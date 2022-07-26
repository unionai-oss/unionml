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

(model)=

# Binding a Model and Dataset

In {ref}`Defining a Dataset <dataset>` we saw how to create a minimal {class}`~unionml.dataset.Dataset`
specification, which uses the sklearn [digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits) and `pandas.DataFrame`
as the underlying data container.

Now let's define a {class}`~unionml.model.Model` and bind it with the {class}`~unionml.dataset.Dataset`.

```{code-block} python
from unionml import DatasetModel

from sklearn.linear_model import LogisticRegression

dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=LogisticRegression, dataset=dataset)
```

```{note}
In the above code snippet you might notice a few things:

- We're defining a `Model` named `"digits_classifier"`.
- The `init` argument is a class, function, or callable object that returns
  your model of interest when called. In this case, we're using the
  `sklearn.linear_model.LogisticRegression` class to train our digits classifier.
- The `dataset` argument takes a `Dataset` object, effectively constraining
  the model's form as a function of the `Dataset` specification.
```

## `Model` Functions

Like the `dataset` object, The `model` object we defined above exposes three
core functions required for model training, prediction, and evaluation.

### {meth}`~unionml.model.Model.init`

In the {class}`~unionml.model.Model` constructor you can note that the `init` argument
takes either a class that is initialized to produce a model object, which will then be passed
down to the `trainer` function as the first positional argument.

In most cases this will suffice, but you can define a decorated `init`
function that achieves the same thing, i.e. the difference is purely syntactic.
The equivalent `init` function would be:

```{code-block} python
@model.init
def init(hyperparameters: dict) -> LogisticRegression:
    return LogisticRegression(**hyperparameters)
```

### {meth}`~unionml.model.Model.trainer`

The `trainer` function should contain all the logic for training a model from
scratch or a previously saved model checkpoint.

```{code-block} python
@model.trainer
def trainer(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())
```

```{note}
The first argument to `trainer` should be the sklearn estimator object that needs to
be updated as a function of the `features` and `target` dataframes.

In this example, the function body simply invokes the sklearn API standard `.fit` method for
training, however you can implement any arbitrary training logic in the `trainer` function.
```

### {meth}`~unionml.model.Model.predictor`

The `predictor` function takes an `estimator` object and `features` dataframe as inputs
and generates a `List` of `float`s representing the predicted digit that the features
represent.

```{code-block} python
from typing import List

@model.predictor
def predictor(estimator: LogisticRegression, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in estimator.predict(features)]
```

### {meth}`~unionml.model.Model.evaluator`

Finally, we need to specify how to evaluate an `estimator` given `features` and a `target`.
In this case, we'll just use the sklearn [`accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) function.

```{code-block} python
from sklearn.metrics import accuracy_score

@model.evaluator
def evaluator(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return accuracy_score(target.squeeze(), predictor(estimator, features))
```

```{note}
Since `predictor` is just a python function, we can use it inside the `evaluator` function body.
```

## Next

Now that we've defined a {class}`~unionml.dataset.Dataset` and {class}`~unionml.model.Model` and bound them
together, let's see how we can perform {ref}`Local Training and Prediction <local_app>`.
