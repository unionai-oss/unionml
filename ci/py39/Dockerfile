FROM python:3.9-slim-buster

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get update && apt-get install -y build-essential git-all

# Install the AWS cli separately to prevent issues with boto being written over
RUN pip3 install awscli

RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
# This assumes that the build context is the root of the unionml repo
COPY ./requirements.txt /root
COPY ./requirements-ci.txt /root
RUN pip install -r /root/requirements-ci.txt

# Copy the actual code
COPY . /root
