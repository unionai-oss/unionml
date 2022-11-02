import typing

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model


@pytest.fixture(scope="function")
def mock_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4] * 25,
            "x2": [1, 2, 3, 4] * 25,
            "x3": [1, 2, 3, 4] * 25,
            "y": [0, 1, 0, 1] * 25,
        }
    )


@pytest.fixture(scope="function", params=[{"custom_init": True}, {"custom_init": False}])
def raw_model(request, mock_data) -> Model:

    dataset = Dataset(
        features=["x"],
        targets=["y"],
        test_size=0.2,
        shuffle=True,
        random_state=123,
    )

    @dataset.reader
    def reader(sample_frac: float, random_state: int) -> pd.DataFrame:
        return mock_data.sample(frac=sample_frac, random_state=random_state)

    @dataset.loader
    def loader(raw_data: pd.DataFrame, head: typing.Optional[int] = None) -> pd.DataFrame:
        if head is not None:
            return raw_data.head(head)
        return raw_data

    model = Model(
        name=f"test_model_{request.param_index}",
        init=None if request.param["custom_init"] else LogisticRegression,
        hyperparameter_config={"C": float, "max_iter": int},
        dataset=dataset,
    )

    if request.param["custom_init"]:
        # define custom init function
        @model.init
        def init_fn(hyperparameters: dict) -> LogisticRegression:
            return LogisticRegression(**hyperparameters)

    return model


@pytest.fixture(scope="function")
def trainer():
    def _trainer(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
        return model.fit(features, target.squeeze())

    return _trainer


@pytest.fixture(scope="function")
def predictor():
    def _predictor(model: LogisticRegression, features: pd.DataFrame) -> typing.List[float]:
        return [float(x) for x in model.predict(features)]

    return _predictor


@pytest.fixture(scope="function")
def evaluator():
    def _evaluator(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
        predictions = model.predict(features)
        return float(accuracy_score(target, predictions))

    return _evaluator


@pytest.fixture(scope="function")
def model(raw_model, trainer, predictor, evaluator):
    raw_model.trainer(trainer)
    raw_model.predictor(predictor)
    raw_model.evaluator(evaluator)
    return raw_model
