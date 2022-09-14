import json
import tempfile
from pathlib import Path
from typing import List
from urllib.parse import unquote_plus

import boto3
import pandas as pd
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
    project="aws-lambda-s3-event",
    domain="development",
)


def lambda_handler(event, context):
    s3_client = boto3.client("s3")  # create s3 client
    model.load_from_env()  # load the model

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        # get features from s3
        with tempfile.NamedTemporaryFile("w") as f:
            s3_client.download_file(bucket, key, f.name)
            features = model.dataset.get_features(Path(f.name))

        # generate prediction
        predictions = model.predict(features=features)

        # upload predictions to s3
        out_filename = "/tmp/predictions.json"
        with open(out_filename, "w") as f:
            json.dump(predictions, f)
        s3_client.upload_file(out_filename, bucket, f"predictions/{key.split('/')[-1]}")


if __name__ == "__main__":
    model_object, metrics = model.train(hyperparameters={"C": 1.0, "max_iter": 10000})
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # save model to a file, using joblib as the default serialization format
    model.save("./model_object.joblib")
