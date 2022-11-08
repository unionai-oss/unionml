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

Scheduling a training job is as easy as adding a {func}`~unionml.schedule.schedule_trainer`
decorator to the function that you want to specify as the {meth}`~unionml.model.Model.trainer`. For example, say that
you're training a `LogisticRegression` model and have the following requirements for your schedule:

- The model must be re-trained at 12am every day
- Use the latest data snapshot as a csv file from some blob store like [AWS S3](https://aws.amazon.com/s3/).

The first thing we need to do is define a {meth}`~unionml.dataset.Dataset.reader` function such that it's able to
get your data based on a time argument:

```{code-cell} python
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression

from unionml import Dataset, Model

dataset = Dataset()
model = Model(dataset=dataset, init=LogisticRegression)


@dataset.reader
def reader(time: datetime) -> pd.DataFrame:
    return pd.read_csv(f"s3://bucket/path/to/dataset/{datetime.strftime('YYYYMMDD')}/dataset.csv")
```

In the `reader` function above, we read the csv file from s3 using [pandas](https://pandas.pydata.org/). Note that
the s3 path containing the dataset conforms to a particular format that depends on the `time` datetime input.

Next, we use the {func}`~unionml.schedule.schedule_trainer` decorator to schedule the function we want to use as the
{meth}`~unionml.model.Model.trainer` component.


```{code-cell} python
from unionml.schedule import schedule_trainer

@model.trainer
@schedule_trainer(
    name="daily_training",
    expression="0 0 * * *",  # train at 8am every day
    reader_time_arg="time",  # feed the schedule kickoff-time `time` to the dataset.reader function
)
def trainer(
    estimator: LogisticRegression,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())
```

We've named the schedule `daily_training`, which is scheduled to run at 12am every morning. The
`reader_time_arg="time"` indicates which argument in the `reader` function we want to use for feeding
in the scheduled kickoff-time, which in turn determines which data is pulled.

```{note}
UnionML doesn't have any opinions about how the data got to the specified s3 path. This is considered to be outside
of the scope of UnionML and assumes that you have some other process or application that handles the ETL process
that produces the data.
```

### Specifying Multiple Schedules

Alternatively, you can specify multiple different kinds of schedules, as long as their names are unique:

```{code-cell} python
from datetime import timedelta
from unionml.schedule import schedule_trainer

@model.trainer
@schedule_trainer(name="cron_schedule", expression="0 0 * * *", reader_time_arg="time")
@schedule_trainer(name="non_standard_schedule", expression="@daily", reader_time_arg="time")
@schedule_trainer(name="fixed_rate_schedule", fixed_rate=timedelta(days=1), reader_time_arg="time")
def trainer(
    estimator: LogisticRegression,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())
```

In the code snippet above we specify daily schedules in three different ways:
1. `cron_schedule`: standard [cron syntax](https://docs.flyte.org/en/latest/concepts/schedules.html#cron-expression)
2. `non_standard_schedule`: non-standard [cronitor syntax](https://github.com/kiorky/croniter#keyword-expressions)
3. `fixed_rate_schedule`: a {class}`~datetime.timedelta` object specifying the job cadence.

### Decorator Ordering

The decoration order doesn't matter: you can also use {func}`~unionml.schedule.schedule_trainer` above the
{meth}`~unionml.model.Model.trainer` decorator:

```{code-cell} python
from datetime import timedelta
from unionml.schedule import schedule_trainer

@schedule_trainer(name="daily_training_above_trainer", expression="0 0 * * *", reader_time_arg="time")
@model.trainer
def trainer(
    estimator: LogisticRegression,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())
```

## Predictor Scheduling

The syntax for specifying batch prediction jobs is similar.

When using batch prediction schedules, you need to ensure that your {meth}`~unionml.dataset.Dataset.reader`
component is correctly factored to account for labeled data (for training and backtesting) and unlabeled data
(for prediction). For example, you can specify a flag that allows you differentiate between training and prediction
settings:

```{code-cell} python
dataset = Dataset()
model = Model(dataset=dataset, init=LogisticRegression)

@dataset.reader
def reader(time: datetime, labeled: bool = True) -> pd.DataFrame:
    uri_prefix = "labeled_datasets" if labeled else "unlabeled_datasets"
    return pd.read_csv(f"s3://bucket/path/to/{uri_prefix}/{datetime.strftime('YYYYMMDD')}/dataset.csv")
```

We've added a `labeled` argument to the `reader` function, which determines the s3 prefix that we use to fetch
the data. Next, we make use of this new argument in the batch prediction schedule via the
{func}`~unionml.schedule.schedule_predictor` decorator:

```{code-cell} python
from typing import List
from unionml.schedule import schedule_predictor

@model.predictor
@schedule_predictor(
    name="daily_predictions",
    expression="0 0 * * *",
    reader_time_arg="time",
    fixed_inputs={"labeled": False},
)
def predictor(estimator: LogisticRegression, features: pd.DataFrame) -> List[int]:
    return [int(x) for x in estimator.predict(features)]
```

Note that we've specified a `fixed_inputs` dictionary, which will be forwarded to the `reader` function
so that it executes correctly and fetches the unlabeled dataset. The `fixed_inputs` argument is one of the
keyword arguments to {class}`flytekit.LaunchPlan`, which is the Flyte construct that is being used under
the hood to create these scheduled jobs.

## Imperative API

The {class}`unionml.model.Model` class exposes an API for adding schedules imperatively:

```{code-cell} python
from unionml.schedule import Schedule

model.add_trainer_schedule(
    Schedule(
        type="trainer",
        name="imperative_training_schedule",
        expression="0 0 * * *",
    )
)

model.add_predictor_schedule(
    Schedule(
        type="predictor",
        name="imperative_predictor_schedule",
        expression="0 0 * * *",
        kwargs={"fixed_inputs": {"labeled": False}},
    )
)
```


## Deploying Schedules to Flyte

Once we're happy with the schedule definitions, we can simply deploy them with
{meth}`unionml.model.Model.remote_deploy` or the
<a href="cli_reference.html#unionml-deploy">unionml deploy</a>
CLI tool.

Both of these options provide a `schedule` flag, which is `True` by default. Specifying
`schedule=false` in the programmatic API or `--no-schedule` in the CLI will disable the deployment
and activation of these scheduled jobs.

Once the schedules are successfully deployed, you can

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
- Deactivate and re-activate them manually.
