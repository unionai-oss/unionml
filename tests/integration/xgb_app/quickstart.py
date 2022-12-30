from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=XGBClassifier, dataset=dataset)

@dataset.reader
def reader() -> pd.DataFrame:
    return load_digits(as_frame=True).frame


@model.trainer
def trainer(estimator: XGBClassifier, features: pd.DataFrame, target: pd.DataFrame) -> XGBClassifier:
    return estimator.fit(features, target.squeeze())


@model.predictor
def predictor(estimator: XGBClassifier, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in estimator.predict(features)]


@model.evaluator
def evaluator(estimator: XGBClassifier, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))


if __name__ == "__main__":
    model_object, metrics = model.train(hyperparameters={'max_depth': 4, 'eta': 0.1, 'sampling_method': 'gradient_based', 'num_class': 3})
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # save model to a file, using joblib as the default serialization format
    model.save("./model_object.joblib")