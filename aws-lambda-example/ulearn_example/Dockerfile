FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./

RUN python3.8 -m pip install -r requirements.txt -t .

# Need to copy the config files to the container
COPY config ./config

COPY app.py ./

# Temporarily install flytekitlearn from the generated wheel.
# Once we release flytekitlearn we can install from PyPI instead of using this locally built wheel.
COPY flytekit_learn-0.0.0+dev0-py3-none-any.whl ./
RUN pip install flytekit_learn-0.0.0+dev0-py3-none-any.whl
RUN pip freeze

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
