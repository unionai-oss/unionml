from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from flytekit_learn import Dataset, Model


dataset = Dataset(
    targets=["target"],
    test_size=0.2,
    shuffle=True,
    random_state=123,
)
model = Model(
    name="breast_cancer",
    init=LogisticRegression,
    hyperparameters={"C": float, "max_iter": int},
    dataset=dataset,
)


@dataset.reader
def reader() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame


@model.trainer
def trainer(model: LogisticRegression, data: List[pd.DataFrame]) -> LogisticRegression:
    features, target = data
    return model.fit(features, target.squeeze())


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

    print("Running flytekit-learn locally")
    breast_cancer_dataset = load_breast_cancer(as_frame=True)
    hyperparameters = {"C": 1.0, "max_iter": 1000}
    trained_model, metrics = model.train(hyperparameters=hyperparameters)
    print(trained_model, metrics)

    predictions = model.predict(trained_model, features=breast_cancer_dataset.frame.sample(5, random_state=42))
    print(predictions)
