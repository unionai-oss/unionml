FROM public.ecr.aws/lambda/python:3.8

RUN yum install -y git

COPY requirements.txt .

RUN python3.8 -m pip install -r requirements.txt -t .
RUN pip list

COPY app.py .

# Copy the model to the Docker image
# NOTE: Make sure to run python app.py before building this image!
COPY model_object.joblib .
ENV UNIONML_MODEL_PATH ./model_object.joblib

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
