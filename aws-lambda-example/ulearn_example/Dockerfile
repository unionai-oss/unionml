FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./

RUN python3.8 -m pip install -r requirements.txt -t .

# Need to copy the config files to the container to be able to start
# flyte remote.
COPY config ./config
COPY app.py ./

# Temporarily install unionml from the generated wheel.
# Once we release unionml we can install from PyPI instead of using this locally built wheel.
COPY unionml-0.0.0+dev0-py3-none-any.whl ./
RUN pip install unionml-0.0.0+dev0-py3-none-any.whl
RUN pip list

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
