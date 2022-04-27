from typing import List

import pandas as pd
from fastapi import FastAPI
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

dataset = Dataset(
    name="breast_cancer_data",
    targets=["target"],
    test_size=0.2,
    shuffle=True,
    random_state=123,
)
model = Model(
    name="breast_cancer_classifier",
    init=LogisticRegression,
    dataset=dataset,
    hyperparameter_config={"C": float, "max_iter": int},
)

# attach Flyte remote backend
model.remote(
    registry="ghcr.io/unionai-oss",
    dockerfile="Dockerfile",
    config_file_path="config/config-remote.yaml",
    project="unionml",
    domain="development",
)

# serve the model with FastAPI
app = FastAPI()
model.serve(app)


@dataset.reader
def reader(sample_frac: float = 1.0, random_state: int = 123) -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame.sample(frac=sample_frac, random_state=random_state)


@model.trainer
def trainer(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
    return model.fit(features, target.squeeze())


@model.predictor
def predictor(model: LogisticRegression, features: pd.DataFrame) -> List[float]:
    """Generate predictions from a model."""
    return [float(x) for x in model.predict_proba(features)[:, 1]]


@model.evaluator
def evaluator(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
    predictions = model.predict(features)
    return float(accuracy_score(target.squeeze(), predictions))


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")

    print("Running unionml locally")
    breast_cancer_dataset = load_breast_cancer(as_frame=True)
    hyperparameters = {"C": 1.0, "max_iter": 1000}
    trained_model, metrics = model.train(hyperparameters, sample_frac=1.0, random_state=123)
    print(trained_model, metrics)

    print("Predicting from reader")
    predictions = model.predict(sample_frac=0.01, random_state=321)
    print(predictions)

    print("Predicting from features")
    predictions = model.predict(features=breast_cancer_dataset.frame.sample(5, random_state=42))
    print(predictions)
