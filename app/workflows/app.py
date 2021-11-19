from typing import Any, Dict, List, Tuple

import os
from flytekit.core.workflow import workflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from flytekit.remote import FlyteRemote
from flytekit_learn import Dataset, Model


breast_cancer = load_breast_cancer()
dataset = Dataset(
    features=list(breast_cancer.feature_names),
    targets=["target"],
    test_size=0.2,
    shuffle=True,
    random_state=100,
)
model = Model(init=LogisticRegression)


@dataset.reader
def reader() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame


@model.trainer
def trainer(model: LogisticRegression, data: List[pd.DataFrame]) -> LogisticRegression:
    features, target = data
    return model.fit(features, target.squeeze())


@model.predictor
def predictor(model: LogisticRegression, features: pd.DataFrame) -> List[float]:
    return model.predict(features)


@model.evaluator
def evaluator(model: LogisticRegression, data: List[pd.DataFrame]) -> float:
    features, target = data
    predictions = model.predict(features)
    return accuracy_score(target, predictions)


train_wf = model.train_workflow(data=dataset())


if __name__ == "__main__":
    trained_model, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        data=dataset(),
    )
    print(trained_model, metrics)
