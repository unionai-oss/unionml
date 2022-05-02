FROM public.ecr.aws/lambda/python:3.8

RUN yum install -y git

COPY requirements.txt .

RUN python3.8 -m pip install -r requirements.txt -t .
RUN pip list

# Need to copy the config files to the container to be able to start
# flyte remote.
COPY config .
COPY app.py .

COPY model_object.joblib .
ENV unionml_MODEL_PATH ./model_object.joblib

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
