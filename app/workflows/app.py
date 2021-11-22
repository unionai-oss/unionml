from typing import Any, Dict, List, Tuple

import os
from flytekit.core.workflow import workflow
import pandas as pd
from fastapi import FastAPI
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from flytekit.remote import FlyteRemote
from flytekit_learn import Dataset, Model


breast_cancer_dataset = load_breast_cancer(as_frame=True)

app = FastAPI()
dataset = Dataset(
    features=list(breast_cancer_dataset.feature_names),
    targets=["target"],
    test_size=0.2,
    shuffle=True,
    random_state=100,
)
model = Model(
    init=LogisticRegression,
    hyperparameters={"C": float, "max_iter": int},
    dataset=dataset,
)
model.serve(app)


@dataset.reader
def reader() -> pd.DataFrame:
    return breast_cancer_dataset.frame


@app.post("/train")
@model.trainer
def trainer(model: LogisticRegression, data: List[pd.DataFrame]) -> LogisticRegression:
    features, target = data
    return model.fit(features, target.squeeze())


@app.get("/predict")
@model.predictor
def predictor(model: LogisticRegression, features: pd.DataFrame) -> List[float]:
    """Generate predictions from a model."""
    return [float(x) for x in model.predict_proba(features)[:, 1]]


@model.evaluator
def evaluator(model: LogisticRegression, data: List[pd.DataFrame]) -> float:
    features, target = data
    predictions = model.predict(features)
    return accuracy_score(target, predictions)


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")

    print("Preparing dataset")
    training_data = breast_cancer_dataset.frame
    features = training_data[breast_cancer_dataset.feature_names]
    prediction_sample = breast_cancer_dataset.frame.sample(5, random_state=42).to_dict(orient="records")

    print("Training with the Literal Dataset")
    hyperparameters = {"C": 1.0, "max_iter": 1000}
    trained_model, metrics = model.train(hyperparameters=hyperparameters, data=training_data)
    print(trained_model, metrics)

    predictions = model.predict(trained_model, features=prediction_sample)
    print(predictions)

    print("Training with a Compiled Dataset Workflow")
    trained_model, metrics = model.train(hyperparameters=hyperparameters, data=dataset())
    print(trained_model, metrics)

    predictions = model.predict(trained_model, features=dataset(features=prediction_sample))
    print(predictions)
