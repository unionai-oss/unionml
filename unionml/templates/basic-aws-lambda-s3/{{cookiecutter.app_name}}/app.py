import json
import logging
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

s3_client = boto3.client("s3")  # create s3 client


def lambda_handler(event, context):
    model.load_from_env()  # load the model

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        # get features from s3
        with tempfile.NamedTemporaryFile("w") as f:
            s3_client.download_file(bucket, key, f.name)
            logger.info("loading features")
            features = model.dataset.get_features(Path(f.name))

        # generate prediction
        predictions = model.predict(features=features)
        logger.info(f"generated predictions {predictions}")

        # upload predictions to s3
        with tempfile.NamedTemporaryFile("w") as out_file:
            json.dump(predictions, out_file)
            upload_key = f"predictions/{key.split('/')[-1]}"
            out_file.flush()
            s3_client.upload_file(out_file.name, bucket, upload_key)
            logger.info(f"uploaded predictions to {bucket}/{upload_key}")


if __name__ == "__main__":
    model_object, metrics = model.train(hyperparameters={"C": 1.0, "max_iter": 10000})
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # save model to a file, using joblib as the default serialization format
    model.save("./model_object.joblib")
