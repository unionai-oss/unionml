FROM public.ecr.aws/lambda/python:3.8

RUN yum install -y git

COPY requirements.txt .

RUN python3.8 -m pip install -r requirements.txt -t .
RUN pip list

COPY app.py .

# Copy a sample model object file and expose it through the env var
COPY model_object.joblib .
# TODO: Change env var to all caps once we update from 0.0.1
ENV UNIONML_MODEL_PATH ./model_object.joblib

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
