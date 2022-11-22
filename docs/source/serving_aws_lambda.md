(serving_aws_lambda)=

# Serving with AWS Lambda

The [Serverless Application Model](https://aws.amazon.com/serverless/sam/) Command Line Interface (SAM CLI) is an extension of the AWS CLI that adds functionality for building
and testing Lambda applications.

It uses Docker to run our functions in an Amazon Linux environment that matches Lambda.
It can also emulate our application's build environment and API locally.

```{admonition} Prerequisites
To follow this guide, we'll need the following tools:

- An [AWS account](https://aws.amazon.com/) for Docker registry authentication.
- SAM CLI: [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html).
- Docker: [Install Docker community edition](https://hub.docker.com/search/?type=edition&offering=community).

We need to use Amazon ECR-based images so be sure to [authenticate to the AWS ECR registry](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-authenticate-registry).
```

## Initialize a UnionML App for AWS Lambda

Initialize a UnionML app that supports serving for AWS Lambda:

```{prompt} bash
:prompts: $

unionml init aws_lambda_app --template basic-aws-lambda
cd aws_lambda_app
```

This will create a UnionML project directory called `aws_lambda_app` which contains
all of the scripts and configuration needed to build and deploy the app.

As we can see in the `app.py` script, the main additions to the UnionML app definition
are that we need to define a `FastAPI` app and wrap it in a `Mangum` object.

```{code-block} python
from fastapi import FastAPI
from mangum import Mangum

# dataset and model definition
dataset = ...
model = ...

# serve with FastAPI
app = FastAPI()
model.serve(app)

# run ASGI applications in AWS Lambda to handle API Gateway using Mangum
lambda_handler = Mangum(app)
```

[Mangum](https://mangum.io/) is an adapter for running ASGI applications in AWS Lambda,
so what we're doing here is using it to convert a `FastAPI` app into an AWS Lambda
serverless function that you can invoke over a web endpoint.


(aws_lambda_build_test_locally)=

## Build and Test Locally

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

Test a single function by invoking it directly with a test *event*. An *event* is a JSON document that represents the input that the function receives from the event source. Test events are included in the `events` folder in this project.

Run functions locally and invoke them with the `sam local invoke` command.

```{prompt} bash
:prompts: $

sam local invoke UnionmlFunction --event events/event.json
```

## Emulating the Lambda API

We can also emulate our application's API:

```{prompt} bash
:prompts: $

sam local start-api
```

Then in another terminal session run the following:

```{prompt} bash
:prompts: $

curl -X POST http://localhost:3000/predict \
    -H "Content-Type: application/json"  \
    -d "{\"features\": $(cat data/sample_features.json)}"
```

````{note}
   SAM CLI reads the application template in `template.yaml` to determine the API's routes
   and the functions that they invoke. The `Events` property on each function's definition
   includes the route and method for each path.

   ```yaml
       Events:
           unionml:
           Type: Api
           Properties:
               Path: /{proxy+}
               Method: post
   ```
````

## Unit Tests

Tests are defined in the `tests` folder in the app directory.

Use pip to install [pytest](https://docs.pytest.org/en/latest/) and run unit tests locally.

```{prompt} bash
:prompts: $

pip install pytest
python -m pytest tests
```

(aws_lambda_deploy)=

## Deploying to AWS Lambda

Once we're satisfied with our application's state, we can then build and deploy it to AWS:

```{note}
If you don't have an account on AWS, create one [here](https://aws.amazon.com/).
```

```{prompt} bash
:prompts: $

sam deploy --guided
```

```{important}
   The first command will build a docker image from a Dockerfile and then copy the source of our application inside the Docker image. The second command will package and deploy our application to AWS, with a series of prompts:

   - **Stack Name**: The name of the stack to deploy to CloudFormation. This should be unique to our
     account and region, and a good starting point would be something matching our project name.
   - **AWS Region**: The AWS region we want to deploy our app to.
   - **Confirm changes before deploy**: If set to yes, any change sets will be shown to we before
     execution for manual review. If set to no, the AWS SAM CLI will automatically deploy application changes.
   - **Allow SAM CLI IAM Role creation**: Many AWS SAM templates, including this example, create AWS IAM
     roles required for the AWS Lambda function(s) included to access AWS services. By default, these are scoped
     down to minimum required permissions. To deploy an AWS CloudFormation stack which creates or modifies IAM
     roles, the `CAPABILITY_IAM` value for `capabilities` must be provided. If permission isn't provided through
     this prompt, to deploy this example we must explicitly pass `--capabilities CAPABILITY_IAM` to the
     `sam deploy` command.
   - **Save arguments to samconfig.toml**: If set to yes, our choices will be saved to a configuration
     file inside  the project, so that in the future we can just re-run `sam deploy` without parameters to
     deploy changes to our application.
```

The output should look something like this:

```{code-block}
...
CloudFormation outputs from deployed stack
----------------------------------------------------------------------------------
Outputs
----------------------------------------------------------------------------------
Key                 UnionmlFunction
Description         unionml Lambda Function ARN
Value               arn:aws:lambda:...

Key                 UnionmlApi
Description         API Gateway endpoint URL for Prod stage for unionml function
Value               https://abcdefghij.execute-api.us-east-42.amazonaws.com/Prod/

Key                 UnionmlFunctionIamRole
Description         Implicit IAM Role created for unionml function
Value               arn:aws:iam::...
----------------------------------------------------------------------------------

Successfully created/updated stack - unionml-example in us-east-2
```

We can find our API Gateway Endpoint URL in the output values displayed after deployment. In this
case, the URL is `https://abcdefghij.execute-api.us-east-42.amazonaws.com/Prod/`.

We can hit that endpoint to generate predictions our unionml app, for example, using the
[`requests`](https://docs.python-requests.org/en/latest/) library:

```python
import requests
from sklearn.datasets import load_digits

digits = load_digits(as_frame=True)
features = digits.frame[digits.feature_names]


prediction_response = requests.post(
    "https://abcdefghij.execute-api.us-east-2.amazonaws.com/Prod/predict",
    json={"features": features.sample(5, random_state=42).to_dict(orient="records")},
)

print(prediction_response.text)
```

### Lambda Function Logs

To simplify troubleshooting, SAM CLI has a command called `sam logs`.

`sam logs` lets we fetch logs generated by our deployed Lambda function from the command line. In addition to printing the logs on the terminal, this command has several nifty features to help we quickly find the bug.

```{prompt} bash
:prompts: $

sam logs -n unionmlFunction --stack-name unionml-aws-lambda-example --tail
```

```{note}
This command works for all AWS Lambda functions, not just the ones we deploy using SAM.
```


### App Resource Cleanup

To delete the sample application that we created, use the AWS CLI. Assuming we used our project name for the stack name, we can run the following:

```{prompt} bash
:prompts: $

sam delete --stack-name unionml-aws-lambda-example
```

### Add a Resource to our Application

The application template uses AWS Serverless Application Model (AWS SAM) to define application resources.
AWS SAM is an extension of AWS CloudFormation with a simpler syntax for configuring common serverless
application resources such as functions, triggers, and APIs.

For resources not included in [the SAM specification](https://github.com/awslabs/serverless-application-model/blob/master/versions/2016-10-31.md), we can use standard [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-template-resource-type-ref.html) resource types.

We can find more information and examples about filtering Lambda function logs in the [SAM CLI Documentation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-logging.html).

### Additional Resources

See the [AWS SAM developer guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html) for an introduction to SAM specification, the SAM CLI, and serverless application concepts.


(serving_aws_lambda_flyte_model)=

## Using a Model Trained on Flyte

So far in this guide we've used a model that we trained locally. But what if we want to us a model
that we trained on a Flyte cluster backend?

The last section of this guide will show you how to do that. Recall that we ran our `app.py` script
like so to generate a model object:

```{prompt} bash
:prompts: $

python app.py
```

We can actually run the same steps that we went through in {ref}`Deploying to Flyte <flyte_cluster>`
to train a model on our Flyte backend:

```{prompt} bash
:prompts: $

flytectl demo start --source .
unionml deploy app:model
unionml train app:model -i '{"hyperparameters": {"C": 1.0, "max_iter": 10000}}'
```

In order to fetch the trained model, you can use `unionml fetch-model` to download and save the model object
to your local directory:

```{prompt} bash
:prompts: $

unionml fetch-model app:model --model-version latest --output-file model_object.joblib
```

This will create a `model_object.joblib` file in your app project directory, which is equivalent
to the model object we created with `python app.py`.

````{note}
Recall that you can list all the current model versions of your app with

   ```{prompt} bash
   :prompts: $

   unionml list-model-versions app:model
   ```

````

From here, you can follow the steps from the {ref}`Build and Test Locally <aws_lambda_build_test_locally>`
or {ref}`Deploy to AWS Lambda <aws_lambda_deploy>` sections to deploy this model to a serverless endpoint!
