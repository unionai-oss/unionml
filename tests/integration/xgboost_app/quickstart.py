from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

# Declare constants and variables at the top of the file
DATASET_NAME = "digits_dataset"
MODEL_NAME = "digits_classifier"
TEST_SIZE = 0.2
SHUFFLE = True
TARGETS = ["target"]
PARAMS = {
    'max_depth': 4, 
    'eta': 0.1, 
    'sampling_method': 'gradient_based', 
    'num_class': 3
}

# Create instances of Dataset and Model
dataset = Dataset(name=DATASET_NAME, test_size=TEST_SIZE, shuffle=SHUFFLE, targets=TARGETS)
model = Model(name=MODEL_NAME, init=XGBClassifier, dataset=dataset)

# Define reader function
@dataset.reader
def reader() -> pd.DataFrame:
    return load_digits(as_frame=True).frame

# Define trainer function
@model.trainer
def trainer(estimator: XGBClassifier, features: pd.DataFrame, target: pd.DataFrame) -> XGBClassifier:
    return estimator.fit(features, target.squeeze())

# Define predictor function
@model.predictor
def predictor(estimator: XGBClassifier, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in estimator.predict(features)]

# Define evaluator function
@model.evaluator
def evaluator(estimator: XGBClassifier, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))

# Main function
def main():
    model_object, metrics = model.train(hyperparameters=PARAMS)
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # save model to a file, using joblib as the default serialization format
    model.save('/tmp/model_object.pickle')

if __name__ == "__main__":
    main()