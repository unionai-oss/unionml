(reacting_to_s3_events)=

# Reacting to S3 Events

In {ref}`serving_aws_lambda` we learned how to deploy a prediction service to an AWS
Lambda function, which we can call as a web endpoint to generate predictions.

But what if you want to generate predictions based on events, like when we upload a
file containing features to an S3 bucket? In this guide, you'll learn how to
build an s3-event-based prediction service.

```{admonition} Prerequisites
To follow this guide, we'll need the following tools:

- An [AWS account](https://aws.amazon.com/) for Docker registry authentication.
- SAM CLI: [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html).
- Docker: [Install Docker community edition](https://hub.docker.com/search/?type=edition&offering=community).

We need to use Amazon ECR-based images so be sure to [authenticate to the AWS ECR registry](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-authenticate-registry).
```

## Initialize a UnionML App for AWS Lambda

```{prompt} bash
:prompts: $

unionml init s3_event_app --template basic-aws-lambda-s3
cd s3_event_app
```

This will create a UnionML project directory called `s3_event_app` which contains
all of the scripts and configuration needed to build and deploy the app.

As we can see in the `app.py` script, the main additions to the UnionML app is the
definition of a `lambda_handler` function that takes in an `event` and `context`
argument.

```{literalinclude} ../../unionml/templates/basic-aws-lambda-s3/{{cookiecutter.app_name}}/app.py
---
lines: 8,50-77
---
```

As you can see, the `lambda_handler` implements the following operations:

- Downloads the features file from s3 and loads it into memory using the {meth}`unionml.dataset.Dataset.get_features` method.
- Generates the prediction using the {meth}`unionml.model.Model.predict` method.
- Uploads the predictions to a `predictions/` prefix in the same s3 bucket. Note that the upload key doesn't have to be the same as the event object key.

```{note}
You can learn more about AWS lambda `event` objects [here](https://docs.aws.amazon.com/lambda/latest/dg/with-s3.html) and `context` objects [here](https://docs.aws.amazon.com/lambda/latest/dg/python-context.html).
```

## Building the App

First we need to create the model object that we want to deploy. we can do this by
simply invoking the `app.py` script:

```{prompt} bash
:prompts: $

python app.py
```

This will create a joblib-serialized sklearn model called `model_object.joblib` in our current directory.

Then, build our application with the `sam build` command.

```{prompt} bash
:prompts: $

sam build
```

The SAM CLI builds a Docker image from the `Dockerfile` and then installs dependencies defined
in `requirements.txt` inside the docker image. The processed template file is saved in the
`.aws-sam/build` folder.

````{note}
   SAM CLI reads the application template in `template.yaml` to determine the s3 resources
   needed for the app, the s3 event trigger definition, and the functions that they invoke.

   The `Resources` property defines the S3 bucket we're going to be reacting to:

   ```{literalinclude} ../../unionml/templates/basic-aws-lambda-s3/{{cookiecutter.app_name}}/template.yaml
   ---
   lines: 8-12,13,45-48
   ---
   ```

   The `Events` property on each function's definition specifies that the lambda function
   should be invoked whenever a `.json` file is uploaded to the `features/` prefix
   of the bucket:

   ```{literalinclude} ../../unionml/templates/basic-aws-lambda-s3/{{cookiecutter.app_name}}/template.yaml
   ---
   lines: 22-34
   ---
   ```

   And finally, the `Policies` key gives the lambda function read and write access to
   the bucket.

   ```{literalinclude} ../../unionml/templates/basic-aws-lambda-s3/{{cookiecutter.app_name}}/template.yaml
   ---
   lines: 35-39
   ---
   ```
````

## Test Locally

You can test the lambda function by running the unit tests, which are in the `tests` folder
in the app directory.

Use pip to install [pytest](https://docs.pytest.org/en/latest/) and run unit tests locally.

```{prompt} bash
:prompts: $

pip install pytest
python -m pytest tests
```

## Deploying to AWS Lambda

Once we're satisfied with our application's state, we can then build and deploy it to AWS:

```{note}
If you don't have an account on AWS, create one [here](https://aws.amazon.com/).
```

```{prompt} bash
:prompts: $

sam deploy --guided
```

The prompt will require you provide inputs to configure your `sam` deployment, which
you can read more about in :ref:`aws_lambda_deploy`. In this example, we'll call the
stack `test-s3-event-unionml-app`.

Once the deployment process is complete, you should see something like this:

```{code-block}
CloudFormation outputs from deployed stack
-----------------------------------------------------------------------------------------------------------------
Outputs
-----------------------------------------------------------------------------------------------------------------
Key                 UnionmlAppBucket
Description         unionml app s3 bucket
Value               arn:aws:s3:::unionml-example-aws-lambda-s3

Key                 UnionmlFunction
Description         unionml Lambda Function ARN
Value               arn:aws:lambda:us-east-2:479331373192:function:test-s3-event-unionml-app-UnionmlFunction-
Vxbl7NiL8Jz7

Key                 UnionmlFunctionIamRole
Description         Implicit IAM Role created for unionml function
Value               arn:aws:iam::479331373192:role/test-s3-event-unionml-app-UnionmlFunctionRole-1LGMQ4OXWD9ZR
-----------------------------------------------------------------------------------------------------------------

Successfully created/updated stack - test-s3-event-unionml-app in us-east-2
```

Where `unionml-example-aws-lambda-s3` is the created bucket, `test-s3-event-unionml-app-UnionmlFunction-
Vxbl7NiL8Jz7`
is the lambda function that's invoked whenever a `.json` file is uploaded to
the `features/` prefix.


### Triggering the S3 Event

The unionml app template ships with some sample features in the `data` directory, which
we can use to trigger the lambda function:

```{prompt} bash
:prompts: $

aws s3 cp data/sample_features.json s3://unionml-example-aws-lambda-s3/features/sample_features-$(date "+%Y%m%d%H%M%S").json
```

Then we can look at the logs with:

```{prompt} bash
:prompts: $

sam logs -n UnionmlFunction --stack-name test-s3-event-unionml-app --tail
```

Finally, let's check whether or not our predictions showed up in the `predictions/`
prefix:

```{prompt} bash
:prompts: $

aws s3 ls s3://unionml-example-aws-lambda-s3/predictions/
```

You should see a file in the format `sample_features-{timestamp}.json`:

```{code-block}
2022-09-16 17:52:46          5 sample_features-20220916175057.json
```

You can download and inspect the contents of the predictions file with:

```{prompt} bash
:prompts: $

aws s3 cp s3://unionml-example-aws-lambda-s3/predictions/sample_features-20220916175057.json .
cat sample_features-20220916175057.json
```

The prediction file should be a json file containing an array of one prediction:

```{code-block}
[8.0]
```

### Invoking the Function Locally

If you want to test the function locally, you can invoke it with `sam local invoke`, with
the events file in `events/event.json`, which is a sample s3 event that you can use to iterate
on the function locally.

However, first you'll need to upload a file to the key `features/sample_features.json`, which is
the file referenced in the `event.json` file.

```{prompt} bash
:prompts: $

aws s3 cp data/sample_features.json s3://unionml-example-aws-lambda-s3/features/
sam local invoke UnionmlFunction --event events/event.json
```

## Summary

Congratulations! 🎉 You just set up an event-based prediction service that invokes
a lambda function whenever you upload files to S3.
