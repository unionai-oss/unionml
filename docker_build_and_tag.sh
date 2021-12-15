#!/bin/bash

set -e

# SET the REGISTRY here, where the docker container should be pushed
REGISTRY="ghcr.io/unionai-oss"

# SET the appname here
APP_NAME="flytekit-learn"

while getopts a:r:v:h flag
do
    case "${flag}" in
        a) APP_NAME=${OPTARG};;
        r) REGISTRY=${OPTARG};;
        v) VERSION=${OPTARG};;
        h) echo "Usage: ${0} [-h|[-a <app_name>][-r <registry_name>][-v <version>]]"
           echo "  h: help (this message)"
           echo "  a: APP_NAME or the REPOSITORY APP_NAME. Defaults to myapp."
           echo "  r = REGISTRY name where the docker container should be pushed. Defaults to none - localhost"
           echo "  v = VERSION of the build. Defaults to using the current git head SHA"
           exit 1;;
        *) echo "Usage: ${0} [-h|[-a <app_name>][-r <registry_name>][-v <version>]]"
           exit 1;;
    esac
done

# If you are using git, then this will automatically use the git head as the
# version
if [ -z "${VERSION}" ]; then
  echo "No version set, using git commit head sha as the version"
  VERSION=$(git rev-parse HEAD)
fi

TAG=${APP_NAME}:${VERSION}
LATEST_TAG=${APP_NAME}:latest
if [ -z "${REGISTRY}" ]; then
  echo "No registry set, creating tag ${TAG}"
else
 TAG="${REGISTRY}/${TAG}"
 LATEST_TAG="${REGISTRY}/${LATEST_TAG}"
 echo "Registry set: creating tags ${TAG}, ${LATEST_TAG}"
fi

# Should be run in the folder that has Dockerfile
docker build --tag ${TAG} --tag ${LATEST_TAG} .
docker push ${TAG}
docker push ${LATEST_TAG}

echo "Docker image built with tag ${TAG} and ${LATEST_TAG}. You can use this image to run pyflyte package."
