import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sklearn.datasets import load_digits

from tests.unit.aws_lambda_app.app import model


@pytest.fixture
def features():
    digits = load_digits(as_frame=True)
    features = digits.frame[digits.feature_names]
    return features.sample(3, random_state=99).to_dict(orient="records")


@pytest.fixture()
def apigw_event(features):
    """Generates API GW Event"""

    return {
        "body": f'{{"features":{json.dumps(features)}}}',
        "resource": "/{proxy+}",
        "requestContext": {
            "resourceId": "123456",
            "apiId": "1234567890",
            "resourcePath": "/{proxy+}",
            "httpMethod": "POST",
            "requestId": "c6af9ac6-7b61-11e6-9a41-93e8deadbeef",
            "accountId": "123456789012",
            "identity": {
                "apiKey": "",
                "userArn": "",
                "cognitoAuthenticationType": "",
                "caller": "",
                "userAgent": "Custom User Agent String",
                "user": "",
                "cognitoIdentityPoolId": "",
                "cognitoIdentityId": "",
                "cognitoAuthenticationProvider": "",
                "sourceIp": "127.0.0.1",
                "accountId": "",
            },
            "stage": "prod",
        },
        "queryStringParameters": {"local": "True"},
        "headers": {
            "Via": "1.1 08f323deadbeefa7af34d5feb414ce27.cloudfront.net (CloudFront)",
            "Accept-Language": "en-US,en;q=0.8",
            "CloudFront-Is-Desktop-Viewer": "true",
            "CloudFront-Is-SmartTV-Viewer": "false",
            "CloudFront-Is-Mobile-Viewer": "false",
            "X-Forwarded-For": "127.0.0.1, 127.0.0.2",
            "CloudFront-Viewer-Country": "US",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Upgrade-Insecure-Requests": "1",
            "X-Forwarded-Port": "443",
            "Host": "1234567890.execute-api.us-east-1.amazonaws.com",
            "X-Forwarded-Proto": "https",
            "X-Amz-Cf-Id": "aaaaaaaaaae3VYQb9jd-nvCd-de396Uhbp027Y2JvkCPNLmGJHqlaA==",
            "CloudFront-Is-Tablet-Viewer": "false",
            "Cache-Control": "max-age=0",
            "User-Agent": "Custom User Agent String",
            "CloudFront-Forwarded-Proto": "https",
            "Accept-Encoding": "gzip, deflate, sdch",
        },
        "pathParameters": {"proxy": "/predict"},
        "httpMethod": "POST",
        "stageVariables": {"baz": "qux"},
        "path": "/predict",
    }


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


def test_aws_api_gateway_lambda_handler(monkeypatch, apigw_event, tmpdir):

    from tests.unit.aws_lambda_app.app import lambda_handler

    model.train(hyperparameters={"C": 1.0, "max_iter": 1000})
    model_path = tmpdir / "model_object.joblib"
    model.save(str(model_path))

    monkeypatch.setenv("UNIONML_MODEL_PATH", str(model_path))

    ret = lambda_handler(apigw_event, "")
    predictions = json.loads(ret["body"])

    assert ret["statusCode"] == 200
    assert predictions == [8.0, 8.0, 0.0]


def test_s3_event_lambda_handler(monkeypatch, features, s3_event, tmpdir, caplog):

    from tests.unit.aws_lambda_app.s3_event import lambda_handler, s3_client

    model.train(hyperparameters={"C": 1.0, "max_iter": 1000})
    model_path = tmpdir / "model_object.joblib"
    model.save(str(model_path))

    monkeypatch.setenv("UNIONML_MODEL_PATH", str(model_path))
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
