import json
import logging
import tempfile
from pathlib import Path
from urllib.parse import unquote_plus

import boto3

from tests.unit.aws_lambda_app.app import model

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
        out_filename = "/tmp/predictions.json"
        with open(out_filename, "w") as f:
            json.dump(predictions, f)

        upload_key = f"predictions/{key.split('/')[-1]}"
        s3_client.upload_file(out_filename, bucket, upload_key)
        logger.info(f"uploaded predictions to {bucket}/{upload_key}")
