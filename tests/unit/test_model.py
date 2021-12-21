import typing
import pytest
from inspect import signature

import pandas as pd
from flytekit.core.python_function_task import PythonFunctionTask
from flytekit.types.pickle import FlytePickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from flytekit_learn import Model, Dataset


@pytest.fixture(scope="function")
def mock_data() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 3, 4] * 25,
        "y": [0, 1, 0, 1] * 25,
    })


@pytest.fixture(scope="function")
def model_def(mock_data) -> Model:

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

    return Model(
        name="test_model",
        init=LogisticRegression,
        hyperparameters={"C": float, "max_iter": int},
        dataset=dataset,
    )


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
        return accuracy_score(target, predictions)
    return _evaluator


@pytest.fixture(scope="function")
def model(model_def, trainer, predictor, evaluator):
    model_def.trainer(trainer)
    model_def.predictor(predictor)
    model_def.evaluator(evaluator)
    return model_def


def test_model_decorators(model, trainer, predictor, evaluator):
    assert model._trainer == trainer
    assert model._predictor == predictor
    assert model._evaluator == evaluator


def test_model_train_task(model, mock_data):
    train_task = model.train_task()
    reader_ret_type = signature(model._dataset._reader).return_annotation
    eval_ret_type = signature(model._evaluator).return_annotation

    assert isinstance(train_task, PythonFunctionTask)
    assert train_task.python_interface.inputs["hyperparameters"] == dict
    assert train_task.python_interface.inputs["data"] == reader_ret_type
    assert train_task.python_interface.outputs["trained_model"].__module__ == "flytekit.types.pickle.pickle"
    assert train_task.python_interface.outputs["metrics"] == typing.Dict[str, eval_ret_type]

    outputs = train_task(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        data=mock_data,
    )

    assert outputs.__class__.__name__ == "TrainingResults"
    assert isinstance(outputs.trained_model, LogisticRegression)
    assert isinstance(outputs.metrics["train"], eval_ret_type)
    assert isinstance(outputs.metrics["test"], eval_ret_type)


def test_model_train(model):
    trained_model, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        sample_frac=1.0,
        random_state=123,
    )
    assert isinstance(trained_model, LogisticRegression)
    assert isinstance(metrics["train"], float)
    assert isinstance(metrics["test"], float)


def test_model_train_from_data(model):
    trained_model, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        sample_frac=1.0,
        random_state=123,
    )
    assert isinstance(trained_model, LogisticRegression)
    assert isinstance(metrics["train"], float)
    assert isinstance(metrics["test"], float)


def test_model_predict_task(model, mock_data):
    predict_task = model.predict_task()
    
    assert isinstance(predict_task, PythonFunctionTask)
    assert predict_task.python_interface.inputs["model"].__module__ == "flytekit.types.pickle.pickle"
    assert predict_task.python_interface.outputs["o0"] == signature(model._predictor).return_annotation

    predictions = predict_task(
        model=LogisticRegression().fit(mock_data[["x"]], mock_data["y"]),
        sample_frac=1.0,
        random_state=123,
    )
    assert all(isinstance(x, float) for x in predictions)


def test_model_predict_from_features_task(model, mock_data):
    predict_from_features_task = model.predict_from_features_task()

    assert isinstance(predict_from_features_task, PythonFunctionTask)
    assert predict_from_features_task.python_interface.inputs["model"].__module__ == "flytekit.types.pickle.pickle"
    assert predict_from_features_task.python_interface.inputs["features"] == signature(
        model._dataset._unpack_features
    ).return_annotation
    assert predict_from_features_task.python_interface.outputs["o0"] == signature(model._predictor).return_annotation

    predictions = predict_from_features_task(
        model=LogisticRegression().fit(mock_data[["x"]], mock_data["y"]),
        features=mock_data[["x"]]
    )
    assert all(isinstance(x, float) for x in predictions)
