import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sklearn.datasets import load_digits


@pytest.fixture()
def s3_event():
    """Generates S3 Event"""

    return {
        "Records": [
            {
                "eventVersion": "2.0",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventTime": "1970-01-01T00:00:00.000Z",
                "eventName": "ObjectCreated:Put",
                "userIdentity": {"principalId": "EXAMPLE"},
                "requestParameters": {"sourceIPAddress": "127.0.0.1"},
                "responseElements": {
                    "x-amz-request-id": "EXAMPLE123456789",
                    "x-amz-id-2": "EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH",
                },
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "configurationId": "testConfigRule",
                    "bucket": {
                        "name": "unionml-example-aws-lambda-s3",
                        "ownerIdentity": {"principalId": "EXAMPLE"},
                        "arn": "arn:aws:s3:::unionml-example-aws-lambda-s3",
                    },
                    "object": {
                        "key": "<REPLACE ME>",  # To be replaced in unit test
                        "size": 1024,
                        "eTag": "0123456789abcdef0123456789abcdef",
                        "sequencer": "0A1B2C3D4E5F678901",
                    },
                },
            }
        ]
    }


@pytest.fixture
def features():
    digits = load_digits(as_frame=True)
    features = digits.frame[digits.feature_names]
    return features.sample(3, random_state=99).to_dict(orient="records")


def test_lambda_handler(monkeypatch, s3_event, features):

    from app import lambda_handler, s3_client

    monkeypatch.setenv("UNIONML_MODEL_PATH", "./tests/unit/model_object.joblib")
    results = {}

    def mock_download_file(bucket, key, filename):
        # This is a nasty piece of state mutation 😈. We basically
        # replace the key with the temporary file created in the
        # lambda handler
        s3_event["Records"][0]["s3"]["object"]["key"] = filename
        with Path(filename).open("w") as f:
            json.dump(features, f)

    def mock_upload_file(filename, bucket, upload_key):
        with open(filename) as f:
            results["output"] = json.load(f)

    with patch.object(s3_client, "download_file", MagicMock(wraps=mock_download_file)) as mock_dl_file, patch.object(
        s3_client, "upload_file", MagicMock(wraps=mock_upload_file)
    ) as mock_ul_file:
        ret = lambda_handler(s3_event, "")

    assert ret is None
    assert results["output"] == [8.0, 8.0, 0.0]
    mock_dl_file.assert_called_once()
    mock_ul_file.assert_called_once()
