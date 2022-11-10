from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=LogisticRegression, dataset=dataset)


@dataset.reader
def reader(time: datetime, labeled: bool = True) -> pd.DataFrame:
    return load_digits(as_frame=True).frame


@model.trainer
def trainer(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
    return estimator.fit(features, target.squeeze())


@model.predictor
def predictor(estimator: LogisticRegression, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in estimator.predict(features)]


@model.evaluator
def evaluator(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))


model.remote(
    dockerfile="ci/py39/Dockerfile",
    config_file=str(Path.home() / ".flyte" / "config.yaml"),
    project="digits-classifier",
    domain="development",
)


if __name__ == "__main__":
    model_object, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 10000}, labeled=False, time=datetime.now()
    )
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # model.schedule_training(
    #     name="daily_training",
    #     expression="*/2 * * * *",  # train every 5 minutes
    #     reader_time_arg="time",  # feed the schedule kickoff-time `time` to the dataset.reader function
    #     labeled=True,
    # )

    model.schedule_prediction(
        name="daily_predictions",
        expression="*/5 * * * *",
        reader_time_arg="time",
        model_object=model_object,
        labeled=False,
    )

    # Limitations:
    # - the model used for prediction must be explicitly specified

    # model.remote_deploy(allow_uncommitted=True, patch=True)
    # model.remote_train(hyperparameters={"C": 1.0, "max_iter": 10000}, time=datetime.now(), labeled=False, wait=True)
    # model.remote_predict(time=datetime.now(), labeled=False)
    model.remote_deactivate_schedules(
        schedule_names=["daily_predictions"], app_version="9784db0bc20817b7bf3c66555e108be00c81cdf4-patch98e6cf6"
    )
