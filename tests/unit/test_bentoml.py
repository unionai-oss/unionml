"""Test BentoML integration."""

import inspect
import typing

import bentoml
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import unionml.services.bentoml
from unionml import Model
from unionml.dataset import FeatureTypeUnion


def test_bentoml_service(model: Model, mock_data: pd.DataFrame):
    """Test that BentoMLService can load a model from multiple sources."""
    bentoml_service = unionml.services.bentoml.BentoMLService(model, framework="sklearn")
    model_object = LogisticRegression()
    model_object.fit(mock_data[["x", "x2", "x3"]], mock_data["y"])

    assert model.artifact is None
    assert bentoml_service.model.artifact is None

    bentoml_model = bentoml_service.save_model(model_object)
    bentoml_service.load_model(bentoml_model.tag.version)
    bentoml_service.configure(
        enable_async=True,
        supported_resources=("cpu",),
        supports_cpu_multi_threading=False,
        runnable_method_kwargs={"batchable": False},
    )

    assert model.artifact is not None
    assert bentoml_service.model.artifact is not None
    assert model.artifact == bentoml_service.model.artifact

    assert isinstance(bentoml_service.svc, bentoml.Service)


class CustomRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        super().__init__()

    @bentoml.Runnable.method
    def predict(self, features):
        ...


@pytest.mark.parametrize("enable_async", [True, False])
def test_create_service(enable_async):
    """Test that a service can be created"""
    runner = bentoml.Runner(CustomRunnable)
    service = unionml.services.bentoml.create_service(
        "test_service",
        runner=runner,
        features=bentoml.io.PandasDataFrame(),
        output=bentoml.io.JSON(),
        enable_async=enable_async,
    )
    assert "predict" in service.apis
    assert isinstance(service.apis["predict"].input, bentoml.io.PandasDataFrame)
    assert isinstance(service.apis["predict"].output, bentoml.io.JSON)

    if enable_async:
        assert inspect.iscoroutinefunction(service.apis["predict"].func)
    else:
        assert not inspect.iscoroutinefunction(service.apis["predict"].func)


@pytest.mark.parametrize("enable_async", [True, False])
@pytest.mark.parametrize("supported_resources", [("cpu",), ("cpu", "nvidia.com/gpu"), ("nvidia.com/gpu",)])
@pytest.mark.parametrize("supports_cpu_multi_threading", [True, False])
@pytest.mark.parametrize(
    "runnable_method_kwargs",
    [
        {"batchable": False},
        {"batchable": True, "batch_dim": 0},
        {"input_spec": bentoml.io.PandasDataFrame()},
        {"output_spec": bentoml.io.JSON()},
    ],
)
def test_create_runnable(enable_async, supported_resources, supports_cpu_multi_threading, runnable_method_kwargs):
    """Test that runner can be created with different settings."""
    runnable = unionml.services.bentoml.create_runnable(
        enable_async=enable_async,
        supported_resources=supported_resources,
        supports_cpu_multi_threading=supports_cpu_multi_threading,
        runnable_method_kwargs=runnable_method_kwargs,
    )
    predict_method = runnable.bentoml_runnable_methods__["predict"]
    if enable_async:
        assert inspect.iscoroutinefunction(predict_method.func)
    else:
        assert not inspect.iscoroutinefunction(predict_method.func)

    assert runnable.SUPPORTED_RESOURCES == supported_resources
    assert runnable.SUPPORTS_CPU_MULTI_THREADING == supports_cpu_multi_threading

    for key in runnable_method_kwargs:
        val = getattr(predict_method.config, key)
        expected = runnable_method_kwargs[key]
        if key == "batch_dim":
            expected = (expected, expected)
        assert val == expected


UNSUPPORTED_TYPES = [int, float, str, bool, set]


@pytest.mark.parametrize(
    "type, expected",
    [
        [FeatureTypeUnion[int, pd.DataFrame], bentoml.io.PandasDataFrame],
        [pd.DataFrame, bentoml.io.PandasDataFrame],
        [np.ndarray, bentoml.io.NumpyNdarray],
        [typing.List, bentoml.io.JSON],
        [typing.List[float], bentoml.io.JSON],
        [typing.List[int], bentoml.io.JSON],
        [typing.List[str], bentoml.io.JSON],
        [typing.Dict, bentoml.io.JSON],
        [typing.Dict[str, float], bentoml.io.JSON],
        [typing.Dict[str, int], bentoml.io.JSON],
        [typing.Dict[str, str], bentoml.io.JSON],
        *[[t, None] for t in UNSUPPORTED_TYPES],
    ],
)
def test_infer_io_descriptor(type, expected):
    result = unionml.services.bentoml.infer_io_descriptor(type)
    assert result == expected


@pytest.mark.parametrize("type", UNSUPPORTED_TYPES)
def test_infer_feature_io_descriptor(type):
    with pytest.raises(TypeError):
        unionml.services.bentoml.infer_feature_io_descriptor(type)


@pytest.mark.parametrize("type", UNSUPPORTED_TYPES)
def test_infer_output_io_descriptor(type):
    with pytest.raises(TypeError):
        unionml.services.bentoml.infer_output_io_descriptor(type)
