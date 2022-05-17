from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI
from mangum import Mangum
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=LogisticRegression, dataset=dataset)


@dataset.reader
def reader() -> pd.DataFrame:
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


# attach Flyte demo cluster metadata
model.remote(
    dockerfile="Dockerfile",
    config_file=str(Path.home() / ".flyte" / "config.yaml"),
    project="{{ cookiecutter.project_name }}",
    domain="development",
)

# serve with FastAPI
app = FastAPI()
model.serve(app)

# run ASGI applications in AWS Lambda to handle API Gateway using Mangum
lambda_handler = Mangum(app)


if __name__ == "__main__":
    model_object, metrics = model.train(hyperparameters={"C": 1.0, "max_iter": 10000})
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # save model to a file, using joblib as the default serialization format
    model.save("./model_object.joblib")
