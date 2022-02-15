import os
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from gradio import networking
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from flytekit_learn import Dataset, Model

dataset = Dataset(
    targets=["target"],
    test_size=0.2,
    shuffle=True,
    random_state=123,
)
model = Model(
    name="breast_cancer",
    init=LogisticRegression,
    hyperparameters={"C": float, "max_iter": int},
    dataset=dataset,
)

# attach Flyte remote backend
model.remote(
    os.environ.get("FLYTE_CONFIG", "config/sandbox.config"),
    project="flytesnacks",
    domain="development",
)

# serve the model with FastAPI
app = FastAPI()
model.serve(app)


@dataset.reader
def reader(sample_frac: float = 1.0, random_state: int = 123) -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame.sample(frac=sample_frac, random_state=random_state)


@model.trainer
def trainer(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
    return model.fit(features, target.squeeze())


@model.predictor
def predictor(model: LogisticRegression, features: pd.DataFrame) -> List[float]:
    """Generate predictions from a model."""
    return [float(x) for x in model.predict_proba(features)[:, 1]]


@model.evaluator
def evaluator(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
    predictions = model.predict(features)
    return accuracy_score(target.squeeze(), predictions)


if __name__ == "__main__":
    local_port = 8000

    # Setup ssh tunnel used to route connections from gradio.app to localhost on the specific
    # local port
    share_url = networking.setup_tunnel(local_server_port=local_port, endpoint=None)
    print(f'\n\nAfter uvicorn server starts, use {share_url} to run your flytekit-learn.\n\n')

    # Start uvicorn server
    uvicorn.run("example.app.main:app", host="127.0.0.1", port=local_port, log_level="info")
