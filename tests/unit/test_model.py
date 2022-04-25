import io
import typing
from inspect import signature

import pandas as pd
import pytest
from flytekit.core.python_function_task import PythonFunctionTask
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from flytekit_learn import Dataset, Model
from flytekit_learn.model import BaseHyperparameters, ModelArtifact


@pytest.fixture(scope="function")
def mock_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4] * 25,
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

    model = Model(
        name="test_model",
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


def test_model_decorators(model, trainer, predictor, evaluator):
    assert model._trainer == trainer
    assert model._predictor == predictor
    assert model._evaluator == evaluator


def test_model_train_task(model, mock_data):
    train_task = model.train_task()
    reader_ret_type = signature(model._dataset._reader).return_annotation
    eval_ret_type = signature(model._evaluator).return_annotation

    assert isinstance(train_task, PythonFunctionTask)
    assert issubclass(train_task.python_interface.inputs["hyperparameters"], BaseHyperparameters)
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


@pytest.mark.parametrize("custom_init", [True, False])
def test_model_train(model, custom_init):
    if custom_init:
        # disable default model initialization
        model._init_cls = None

        # define custom init function
        @model.init
        def init(hyperparameters: dict) -> LogisticRegression:
            return LogisticRegression(**hyperparameters)

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

    trained_model = LogisticRegression().fit(mock_data[["x"]], mock_data["y"])
    predictions = predict_task(model=trained_model, data=mock_data[["x"]])

    model.artifact = ModelArtifact(trained_model)
    alt_predictions = model.predict(features=mock_data[["x"]])

    assert all(isinstance(x, float) for x in predictions)
    assert predictions == alt_predictions


def test_model_predict_from_features_task(model, mock_data):
    predict_from_features_task = model.predict_from_features_task()

    assert isinstance(predict_from_features_task, PythonFunctionTask)
    assert predict_from_features_task.python_interface.inputs["model"].__module__ == "flytekit.types.pickle.pickle"
    assert (
        predict_from_features_task.python_interface.inputs["features"]
        == signature(model._dataset._feature_getter).return_annotation
    )
    assert predict_from_features_task.python_interface.outputs["o0"] == signature(model._predictor).return_annotation

    predictions = predict_from_features_task(
        model=LogisticRegression().fit(mock_data[["x"]], mock_data["y"]),
        features=mock_data[["x"]],
    )
    assert all(isinstance(x, float) for x in predictions)


def test_model_saver_and_loader_filepath(model, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_obj, _ = model.train(hyperparameters={"C": 1.0, "max_iter": 1000}, sample_frac=1.0, random_state=42)
    output_path, *_ = model.save(model_path)

    assert output_path == str(model_path)

    loaded_model_obj = model.load(output_path)
    assert model_obj.get_params() == loaded_model_obj.get_params()


def test_model_saver_and_loader_fileobj(model):
    fileobj = io.BytesIO()
    model_obj, _ = model.train(hyperparameters={"C": 1.0, "max_iter": 1000}, sample_frac=1.0, random_state=42)
    model.save(fileobj)
    loaded_model_obj = model.load(fileobj)
    assert model_obj.get_params() == loaded_model_obj.get_params()
