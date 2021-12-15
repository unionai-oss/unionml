# flytekit-learn

The easiest way to build and deploy models

Run the app locally:

```bash
python example/workflows/app.py
```

Start the FastAPI server:

```bash
uvicorn example.app.main:app --reload
```

In a different shell session, make calls to the API that run locally

```bash
python example/requests/local.py  # using a python script via the requests library
source example/requests/local.sh  # or a shell script via curl
```

# Sandbox Cluster

Setup:

```bash
flytectl sandbox start --source .
flytectl sandbox exec -- docker build . --tag "flytekit-learn:v0"  # build app container on sandbox
```

Deploy the model:

```bash
fklearn deploy example.app.main:model -i "flytekit-learn:v0" -v 0  # deploy flytekit-learn model
```

Train model on a Flyte backend

```bash
fklearn train example.app.main:model -i '{"hyperparameters": {"C": 1.0, "max_iter": 1000}, "sample_frac": 1.0, "random_state": 123}'
```

Generate predictions:

```bash
fklearn predict example.app.main:model -i '{"sample_frac": 0.01, "random_state": 123}'  # from the reader
fklearn predict example.app.main:model -f example/data/sample_breast_cancer_data.json  # or with json data
```

Call the Flyte backend via the FastAPI app:

```bash
python example/requests/remote.py  # using a python script via the requests library
source example/requests/remote.sh  # or a shell script via curl
```

# Remote Cluster

Setup:

```bash
export FLYTE_CONFIG=config/remote.config  # point to the remote cluster
export VERSION=v1
./docker_build_and_tag.sh  # build and push your docker image
```

Start the FastAPI server:

```bash
uvicorn app.workflows.app:app --reload
```

Deploy the model:

```bash
fklearn deploy example.app.main:model -i "ghcr.io/unionai-oss/flytekit-learn:$VERSION" -v $VERSION
```

To run CLI commands or FastAPI endpoint calls against the remote Flyte backend, you'll need
to have the correct credentials in your `~/.aws/credentials` file, or you can export the following aws
credentials to you environment:

```
export AWS_ACCESS_KEY_ID="<REPLACE_WITH_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<REPLACE_WITH_SECRET_KEY>"
export AWS_SESSION_TOKEN="<REPLACE_WITH_SESSION_TOKEN>"
```

Train model on a Flyte backend

```bash
fklearn train example.app.main:model -i '{"hyperparameters": {"C": 1.0, "max_iter": 1000}, "sample_frac": 1.0, "random_state": 123}'
```

Generate predictions from reader:

```bash
fklearn predict example.app.main:model -i '{"sample_frac": 0.01, "random_state": 123}'  # from the reader
fklearn predict example.app.main:model -f example/data/sample_breast_cancer_data.json  # or with json data
```
