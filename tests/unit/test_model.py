"""Test UnionML Model object.

Fixtures are defined in conftest.py
"""

import io
import typing
from datetime import timedelta
from inspect import signature

import pandas as pd
import pytest
from flytekit import task, workflow
from flytekit.core.python_function_task import PythonFunctionTask
from sklearn.linear_model import LogisticRegression

from unionml.model import BaseHyperparameters, Model, ModelArtifact
from unionml.schedule import Schedule, ScheduleType


def test_model_decorators(model: Model, trainer, predictor, evaluator):
    assert model._trainer == trainer
    assert model._predictor == predictor
    assert model._evaluator == evaluator


def test_model_train_task(model: Model, mock_data: pd.DataFrame):
    train_task = model.train_task()

    assert model._dataset._reader is not None
    assert model._evaluator is not None

    reader_ret_type = signature(model._dataset._reader).return_annotation
    eval_ret_type = typing.cast(typing.Type, signature(model._evaluator).return_annotation)

    assert isinstance(train_task, PythonFunctionTask)
    assert issubclass(train_task.python_interface.inputs["hyperparameters"], BaseHyperparameters)
    assert train_task.python_interface.inputs["data"] == reader_ret_type
    assert train_task.python_interface.outputs["model_object"].__module__ == "flytekit.types.pickle.pickle"
    assert train_task.python_interface.outputs["metrics"] == typing.Dict[str, eval_ret_type]  # type: ignore

    outputs = train_task(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        data=mock_data,
    )

    assert outputs.__class__.__name__ == "ModelArtifact"
    assert isinstance(outputs.model_object, LogisticRegression)
    assert isinstance(outputs.metrics["train"], eval_ret_type)
    assert isinstance(outputs.metrics["test"], eval_ret_type)


@pytest.mark.parametrize("custom_init", [True, False])
def test_model_train(model: Model, custom_init):
    if custom_init:
        # disable default model initialization
        model._init_cls = None

        # define custom init function
        @model.init
        def init(hyperparameters: dict) -> LogisticRegression:
            return LogisticRegression(**hyperparameters)

    model_object, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        sample_frac=1.0,
        random_state=123,
    )
    assert isinstance(model_object, LogisticRegression)
    assert isinstance(metrics["train"], float)
    assert isinstance(metrics["test"], float)


@pytest.mark.parametrize(
    "dataset_kwargs",
    [
        {},
        {"loader_kwargs": {"head": 20}},
        {"splitter_kwargs": {"test_size": 0.5, "shuffle": False, "random_state": 54321}},
        {"parser_kwargs": {"features": ["x2", "x3"], "targets": ["y"]}},
    ],
)
def test_model_train_from_data(model: Model, dataset_kwargs):
    model_object, metrics = model.train(
        hyperparameters={"C": 1.0, "max_iter": 1000},
        sample_frac=1.0,
        random_state=123,
        **dataset_kwargs,
    )
    assert isinstance(model_object, LogisticRegression)
    assert isinstance(metrics["train"], float)
    assert isinstance(metrics["test"], float)


def test_model_predict_task(model: Model, mock_data: pd.DataFrame):
    predict_task = model.predict_task()

    assert isinstance(predict_task, PythonFunctionTask)
    assert predict_task.python_interface.inputs["model_object"].__module__ == "flytekit.types.pickle.pickle"
    assert predict_task.python_interface.outputs["o0"] == signature(model._predictor).return_annotation

    model_object = LogisticRegression().fit(mock_data[["x"]], mock_data["y"])
    predictions = predict_task(model_object=model_object, data=mock_data[["x"]])

    model.artifact = ModelArtifact(model_object)
    alt_predictions = model.predict(features=mock_data[["x"]])

    assert all(isinstance(x, float) for x in predictions)
    assert predictions == alt_predictions


def test_model_predict_from_features_task(model: Model, mock_data: pd.DataFrame):
    predict_from_features_task = model.predict_from_features_task()

    assert model._dataset._reader is not None
    assert isinstance(predict_from_features_task, PythonFunctionTask)
    assert (
        predict_from_features_task.python_interface.inputs["model_object"].__module__ == "flytekit.types.pickle.pickle"
    )
    assert (
        predict_from_features_task.python_interface.inputs["features"]
        == signature(model._dataset._reader).return_annotation
    )
    assert predict_from_features_task.python_interface.outputs["o0"] == signature(model._predictor).return_annotation

    predictions = predict_from_features_task(
        model_object=LogisticRegression().fit(mock_data[["x"]], mock_data["y"]),
        features=mock_data[["x"]],
    )
    assert all(isinstance(x, float) for x in predictions)


def test_model_saver_and_loader_filepath(model: Model, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_obj, _ = model.train(hyperparameters={"C": 1.0, "max_iter": 1000}, sample_frac=1.0, random_state=42)
    output_path, *_ = model.save(model_path)

    assert output_path == str(model_path)

    loaded_model_obj = model.load(output_path)
    assert model_obj.get_params() == loaded_model_obj.get_params()


def test_model_saver_and_loader_fileobj(model: Model):
    fileobj = io.BytesIO()
    model_obj, _ = model.train(hyperparameters={"C": 1.0, "max_iter": 1000}, sample_frac=1.0, random_state=42)
    model.save(fileobj)
    loaded_model_obj = model.load(fileobj)
    assert model_obj.get_params() == loaded_model_obj.get_params()


def test_model_train_task_in_flyte_workflow(model: Model, mock_data: pd.DataFrame):
    """Test that the unionml.Model-derived training task can be used in regular Flyte workflows."""

    ModelInternals = typing.NamedTuple("ModelInternals", [("coef", typing.List[float]), ("intercept", float)])

    train_task = model.train_task()

    @task
    def get_model_internals(model_object: LogisticRegression) -> ModelInternals:
        """Task that gets coefficients and biases of the model."""
        return ModelInternals(coef=model_object.coef_[0].tolist(), intercept=model_object.intercept_.tolist()[0])

    @workflow
    def wf(data: pd.DataFrame) -> ModelInternals:
        model_artifact = train_task(
            hyperparameters={"C": 1.0, "max_iter": 1000},
            data=data,
            loader_kwargs={},
            splitter_kwargs={},
            parser_kwargs={},
        )
        return get_model_internals(model_object=model_artifact.model_object)

    output = wf(data=mock_data)
    assert isinstance(output.coef, list)
    assert all(isinstance(x, float) for x in output.coef)
    assert isinstance(output.intercept, float)


def test_model_predict_task_in_flyte_workflow(model: Model, mock_data: pd.DataFrame):
    """Test that the unionml.Model-derived prediction task can be used in regular Flyte workflows."""
    model_obj = LogisticRegression()
    model_obj.fit(mock_data[["x", "x2", "x3"]], mock_data["y"])

    predict_task = model.predict_task()

    @task
    def normalize_predictions(predictions: typing.List[float]) -> typing.List[float]:
        """Task that normalizes predictions."""
        s = pd.Series(predictions)
        return (s - s.mean() / s.std()).tolist()

    @workflow
    def wf(model_obj: LogisticRegression, features: pd.DataFrame) -> typing.List[float]:
        predictions = predict_task(model_object=model_obj, data=features)
        return normalize_predictions(predictions=predictions)

    normalized_predictions = wf(model_obj=model_obj, features=mock_data[["x", "x2", "x3"]])

    assert all(isinstance(x, float) for x in normalized_predictions)
    assert any(x < 0 for x in normalized_predictions)
    assert any(x > 0 for x in normalized_predictions)


def test_model_schedule(model: Model):
    """Test that scheduling multiple models with different names will product an error."""
    expression = "0 * * * *"
    fixed_rate = timedelta(days=1)

    model.add_training_schedule(
        Schedule(ScheduleType.trainer, name=f"{model.name}_training_schedule_expression", expression=expression)
    )
    model.add_training_schedule(
        Schedule(ScheduleType.trainer, name=f"{model.name}_training_schedule_fixed_rate", fixed_rate=fixed_rate)
    )

    model.add_prediction_schedule(
        Schedule(ScheduleType.predictor, name=f"{model.name}_prediction_schedule_expression", expression=expression)
    )
    model.add_prediction_schedule(
        Schedule(ScheduleType.predictor, name=f"{model.name}_prediction_schedule_fixed_rate", fixed_rate=fixed_rate)
    )

    assert len(model.training_schedule_names) == 2
    assert len(model.prediction_schedule_names) == 2
    assert len(model.training_schedules) == 2
    assert len(model.prediction_schedules) == 2
    assert all(isinstance(x, Schedule) for x in model.training_schedules)
    assert all(isinstance(x, Schedule) for x in model.prediction_schedules)
    assert set(model.training_schedule_names) == set(
        f"{model.name}_{x}" for x in ("training_schedule_expression", "training_schedule_fixed_rate")
    )
    assert set(model.prediction_schedule_names) == set(
        f"{model.name}_{x}" for x in ("prediction_schedule_expression", "prediction_schedule_fixed_rate")
    )
