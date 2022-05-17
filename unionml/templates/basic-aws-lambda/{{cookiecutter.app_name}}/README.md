# Basic AWS Lambda Example

This project contains source code and supporting files for a serverless application that you can deploy with the SAM CLI. It includes the following files and folders:

- `app.py`: The UnionML app code for digits classification.
- `Dockerfile`: Dockerfile to package up the UnionML app for remote backend deployment.
- `Dockerfile.awslambda`: Dockerfile to package up the UnionML prediction endpoint for AWS Lambda serving.
- `events`: Invocation events that you can use to invoke the function.
- `data`: Sample data used for invoking prediction endpoints.
- `tests`: Unit tests for the application code.
- `template.yaml`: A template that defines the application's AWS resources.

The application uses several AWS resources, including Lambda functions and an API Gateway API. These resources are defined in the `template.yaml` file in this project. You can update the template to add AWS resources through the same deployment process that updates your application code.

To learn more about how to setup, test, and deploy a UnionML app to AWS Lambda,
check out the 📖 [Documentation](https://unionml.readthedocs.io/en/latest/deploying.html).
