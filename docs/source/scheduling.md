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

(scheduling)=

# Scheduling

UnionML enables you to schedule training and prediction jobs on a Flyte cluster so that it runs at a particular time
or on a particular cadence.

```{admonition} Prerequisites
:class: important

- Understand the {ref}`basics <basics>` of how to create a UnionML app.
- Spin up a {ref}`Flyte Cluster <flyte_cluster>`, which will execute the scheduled jobs.
```

## Trainer Scheduling

Scheduling a training job is as easy as invoking the {func}`~unionml.model.Model.schedule_training`
method after you've defined all of your UnionML app components. For example, say that
you're training a `LogisticRegression` model and have the following requirements for your schedule:

- The model must be re-trained at 12am every day
- Use the latest data snapshot as a csv file from some blob store like [AWS S3](https://aws.amazon.com/s3/).

The first thing we need to do is define the minimum components that you need to implement for a UnionML
app (assuming you're working with ``pandas.DataFrame`` objects), namely the
{meth}`~unionml.dataset.Dataset.reader`, {meth}`~unionml.model.Model.trainer`,
{meth}`~unionml.model.Model.predictor`, and {meth}`~unionml.model.Model.evaluator` functions.

```{code-cell} python
from datetime import datetime
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

dataset = Dataset(targets=["target"])
model = Model(dataset=dataset, init=LogisticRegression)


@dataset.reader
def reader(time: datetime) -> pd.DataFrame:
    return pd.read_csv(
        f"s3://bucket/path/to/dataset/{time.strftime('YYYYMMDD')}/dataset.csv"
    )

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
    features: pd.DataFrame,
) -> List[int]:
    return [int(x) for x in estimator.predict(features)]


@model.evaluator
def evaluator(
    estimator: LogisticRegression,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> float:
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))
```

In the `reader` function above, we read the csv file from s3 using [pandas](https://pandas.pydata.org/). Note that
the s3 path containing the dataset conforms to a particular format that depends on the `time` datetime input.

Next, we use the {meth}`~unionml.model.Model.schedule_training` method to lazily configure a training schedule
to the app, which will be deployed to the remote {ref}`Flyte Cluster <flyte_cluster>` in the app deployment step
(we'll get to that later in this guide 👇).


```{code-cell} python
model.schedule_training(
    name="cron_daily_training",
    expression="0 0 * * *",  # train at 12am every day
    reader_time_arg="time",  # feed the schedule kickoff-time `time` to the dataset.reader function
    hyperparameters={"C": 0.1},
)
```

We've named the schedule `daily_training`, which is scheduled to run at 12am every morning. The
`reader_time_arg="time"` indicates which argument in the `reader` function we want to use for feeding
in the scheduled kickoff-time, which in turn determines which data is pulled. We also feed in additional
hyperparameters that we want to use to train the model.

```{note}
UnionML doesn't have any opinions about how the data got to the specified s3 path. This is considered to be
outside of the scope of UnionML and assumes that you have some other process or application that handles the ETL
process that produces the data.
```

### Specifying Multiple Schedules

Alternatively, you can specify multiple different kinds of schedules, as long as their names are unique:

```{code-cell} python
from datetime import timedelta

# equivalent to the above schedule, except here use the non-standard croniter syntax.
model.schedule_training(
    name="non_standard_training_schedule",
    expression="@daily",
    reader_time_arg="time",
    hyperparameters={"C": 0.1},
)

# run the schedule with a fixed rate, if you don't care about the schedule running at
# a particular time of day.
model.schedule_training(
    name="fixed_rate_training_schedule",
    fixed_rate=timedelta(days=1),
    reader_time_arg="time",
    hyperparameters={"C": 0.1},
)
```

In the two code snippets above we specify daily schedules in three different ways:
1. `cron_daily_training`: standard [cron syntax](https://docs.flyte.org/en/latest/concepts/schedules.html#cron-expression)
2. `non_standard_training_schedule`: non-standard [cronitor syntax](https://github.com/kiorky/croniter#keyword-expressions)
3. `fixed_rate_training_schedule`: a {class}`~datetime.timedelta` object specifying the job cadence.

## Predictor Scheduling

The syntax for specifying batch prediction jobs is similar.

When using batch prediction schedules, you need to ensure that your {meth}`~unionml.dataset.Dataset.reader`
component is correctly factored to account for labeled data (for training and backtesting) and unlabeled data
(for prediction). For example, you can specify a flag that allows you to differentiate between training and prediction settings:

```{code-cell} python
:tags: [remove-cell]

# monkey-patch read_csv so it returns data
from sklearn.datasets import load_digits

pd.read_csv = lambda *args, **kwargs: load_digits(as_frame=True).frame
```

```{code-cell} python
@dataset.reader
def reader(time: datetime, labeled: bool = True) -> pd.DataFrame:
    uri_prefix = "labeled_datasets" if labeled else "unlabeled_datasets"
    return pd.read_csv(f"s3://bucket/path/to/{uri_prefix}/{time.strftime('YYYYMMDD')}/dataset.csv")
```

We've added a `labeled` argument to the `reader` function, which determines the s3 prefix that we use to fetch
the data. Next, we make use of this new argument in the batch prediction schedule via the
{meth}`~unionml.model.Model.schedule_prediction` method.

First, however, we need to train a model locally so that the schedule can use a specific model object. Calling
{meth}`~unionml.model.Model.train` will assign the {attr}`~unionml.model.Model.artifact` property, which
{meth}`~unionml.model.Model.schedule_prediction` will use under the hood.

```{code-cell} python
from datetime import datetime
from typing import List

# train a model locally
model.train(hyperparameters={"C": 0.1, "max_iter": 5000}, time=datetime.now(), labeled=True)

# this will use the model object in the model.artifact property
model.schedule_prediction(
    name="daily_predictions",
    expression="0 0 * * *",
    reader_time_arg="time",
    labeled=False,
)
```

Note that we specify a `labeled` keyword argument to indicate that we're working with unlabeled data.
Just like the {meth}`~unionml.model.Model.train`, {meth}`~unionml.model.Model.predict`,
{meth}`~unionml.model.Model.remote_train`, {meth}`~unionml.model.Model.remote_predict` methods, the
scheduling methods take in the `reader` function arguments as keyword arguments.


### Specifying Models in Multiple Ways

You can specify the model you want to use for batch prediction in three other ways.

**Explicitly Passing in a `model_object`**

```{code-cell} python
model_object, _ = model.train(hyperparameters={"C": 0.1}, time=datetime.now(), labeled=True)

model.schedule_prediction(
    name="daily_predictions_with_model_object",
    expression="0 0 * * *",
    reader_time_arg="time",
    model_object=model_object,
    labeled=False,
)
```

Note that `model_object` could have originated from anywhere, not just from `model.train`,
as long as the model type is consistent with the type specified in {meth}`~unionml.model.Model.trainer`.

**Passing in a filepath to a serialized model object**

If you have a serialized model object in a file, you can pass in the file path through
the `model_file` argument. As with the `model_object` option, the underlying model object
in the serialized file must be the same model type as defined in the {meth}`~unionml.model.Model.trainer`.

You may also pass in additional keyword arguments via `loader_kwargs`, which will be forwarded
to that model type's {meth}`~unionml.model.Model.loader` component. See the
{meth}`~unionml.model.Model._default_loader` and {meth}`~unionml.model.Model._default_saver`
 methods for more details.

```{code-cell} python
from tempfile import NamedTemporaryFile


with NamedTemporaryFile() as f:
    # saves model.artifact.model_object to a file, forwarding kwargs to joblib.dump
    model.save(f.name, compress=3)

    # schedule prediction using the model object in the file, forward kwargs to joblib.load
    model.schedule_prediction(
        name="daily_predictions_with_model_file",
        expression="0 0 * * *",
        reader_time_arg="time",
        model_file=f.name,
        loader_kwargs={"mmap_mode": None},
        labeled=False,
    )
```

```{note}
The only supported model types, by default, are `sklearn`, `pytorch`, and `keras` models. If you're
working with a different model type, you'll need to implement the {meth}`~unionml.model.Model.saver`
and {meth}`~unionml.model.Model.loader` components.
```

**Referencing a `model_version`**

You can use the {meth}`~unionml.model.Model.remote_list_model_versions` methods to get version string
identifiers of models that you previously trained on a {ref}`Flyte Cluster <flyte_cluster>`. In the case
that your model version isn't associated with the latest version of your UnionML app, you'll
also need to pass in an `app_version` argument, which is the output of invoking
{meth}`~unionml.model.Model.remote_deploy`. The app version can also be found in the Flyte console UI,
which is essentially the version string associated with any of the workflows that UnionML registers
for you when you deploy your app.

```{code-block} python
model.schedule_prediction(
    name="daily_predictions_with_remote_model",
    expression="0 0 * * *",
    reader_time_arg="time",
    model_version="<MODEL_VERSION>",
    app_version="<APP_VERSION>",
    loader_kwargs={},
    labeled=False,
)
```

## Deploying Schedules to Flyte

Once we're happy with the schedule definitions, we can simply deploy them with
{meth}`unionml.model.Model.remote_deploy` or the
<a href="cli_reference.html#unionml-deploy">unionml deploy</a>
CLI tool.

```{code-block} python
model.remote(project="my-unionml-app", domain="development")
model.remote_deploy(schedule=True)
```

```{note}
Make sure you follow the {ref}`flyte_cluster` guide to understand how deployment works.
```

Both of these options provide a `schedule` flag, which is `True` by default. Specifying
`schedule=false` in the programmatic API or `--no-schedule` in the CLI will disable the deployment
and activation of these scheduled jobs.

Once the schedules are successfully deployed, you can go to the
[Flyte Console UI](https://docs.flyte.org/en/latest/concepts/console.html#divedeep-console) to check out
their execution status and progress. You can also inspect and fetch them programmatically.
For the training schedules, you can do something like:

```{code-block} python
from unionml import ModelArtifact

# get the latest FlyteWorkflowExecutions associated with training schedule runs
latest_training_execution, *_ = model.remote_list_scheduled_training_runs("daily_training")

# fetch the latest model artifact
model_artifact: ModelArtifact = model.remote_fetch_model(latest_training_execution)
model_object = model_artifact.model_object

# use the model_object for some downstream purpose
...
```

And something similar for prediction schedules

```{code-block} python

# get the latest FlyteWorkflowExecution associated with prediction schedule runs
latest_prediction_execution, *_ = model.remote_list_scheduled_prediction_runs("daily_predictions")

# fetch the latest prediction
predictions = model.remote_fetch_predictions(latest_prediction_execution)

# use the predictions for some downstream purpose
...
```

### Manually Activating and Deactivating Schedules

By default, deploying the UnionML App will deploy and activate all the declared schedules. You can manually
deactivate and re-activate them by using the programmatic API:

- {meth}`unionml.model.Model.remote_deactivate_schedules`
- {meth}`unionml.model.Model.remote_activate_schedules`

Or the CLI tool:

- <a href="cli_reference.html#unionml-activate-schedules">activate-schedules</a>
- <a href="cli_reference.html#unionml-deactivate-schedules">deactivate-schedules</a>

## Summary

In this guide, you learned how to:
- Schedule training and prediction jobs.
- Deploy them to a Flyte cluster.
- Get the metadata and outputs of executed schedules.
- Deactivate and re-activate them manually.
