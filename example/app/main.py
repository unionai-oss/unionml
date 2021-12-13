import os
import math
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
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
model.remote(os.environ.get("FLYTE_CONFIG", "config/sandbox.config"), project="flytesnacks", domain="development")

# serve the model with FastAPI
app = FastAPI()
model.serve(app)


@dataset.reader
def reader(sample_frac: float = 1.0, random_state: int = 123) -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame.sample(frac=sample_frac, random_state=random_state)


# NOTE: it probably makes more sense to create a special Labeller class embedded within a Dataset
# object, which exposes various decorator endpoints to achieve different things, such as:
# - yielding a batch of data (this could already be done by Dataset.iterator)
# - creating a labelling session
# - submitting labels
@app.post("/label/{session_id}")
@dataset.labeller
def labeller(session_id: int, batch_size: int, state: dict = None, **reader_kwargs):
    data = reader(**reader_kwargs)
    n_batches = math.ceil(data.shape[0] / batch_size)
    for batch in np.array_split(data, n_batches):
        labelled_batch = yield batch.to_dict(orient="records")
        print(f"labelled batch: {labelled_batch}")


@app.post("/train")
@model.trainer
def trainer(model: LogisticRegression, data: List[pd.DataFrame]) -> LogisticRegression:
    features, target = data
    return model.fit(features, target.squeeze())


@app.get("/predict")
@model.predictor
def predictor(model: LogisticRegression, features: pd.DataFrame) -> List[float]:
    """Generate predictions from a model."""
    return [float(x) for x in model.predict_proba(features)[:, 1]]


@model.evaluator
def evaluator(model: LogisticRegression, data: List[pd.DataFrame]) -> float:
    features, target = data
    predictions = model.predict(features)
    return accuracy_score(target, predictions)


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")

    print("Running flytekit-learn locally")
    breast_cancer_dataset = load_breast_cancer(as_frame=True)
    hyperparameters = {"C": 1.0, "max_iter": 1000}
    trained_model, metrics = model.train(hyperparameters, sample_frac=1.0, random_state=123)
    print(trained_model, metrics)

    print("Predicting from reader")
    predictions = model.predict(trained_model, sample_frac=0.01, random_state=321)
    print(predictions)

    print("Predicting from features")
    predictions = model.predict(trained_model, features=breast_cancer_dataset.frame.sample(5, random_state=42))
    print(predictions)
