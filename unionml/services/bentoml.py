"""Service definition for BentoML."""

import typing
from pathlib import Path

try:
    from typing import get_args, get_origin  # type: ignore
except ImportError:
    from typing_extensions import get_args, get_origin

import bentoml
import numpy as np
import pandas as pd

from unionml.dataset import R
from unionml.model import Model, resolve_model_artifact
from unionml.services.base import Service

if typing.TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod

    class RunnerImpl(bentoml.Runner):
        predict: RunnerMethod


class BentoMLService(Service):
    def __init__(
        self,
        model: Model,
        features: typing.Optional[bentoml.io.IODescriptor] = None,
        output: typing.Optional[bentoml.io.IODescriptor] = None,
    ):
        super().__init__()
        self._model: Model = model
        self._features = features
        self._output = output
        self._svc = None

    @property
    def svc(self) -> bentoml.Service:
        if self._svc is not None:
            return self._svc
        self._svc = self._define_api(self._features, self._output)
        return self._svc

    @property
    def model(self) -> Model:
        return self._model

    def _define_api(
        self,
        features: typing.Optional[bentoml.io.IODescriptor] = None,
        output: typing.Optional[bentoml.io.IODescriptor] = None,
    ) -> bentoml.Service:
        runner = typing.cast(
            "RunnerImpl",
            bentoml.Runner(
                UnionMLRunner,
                name=f"unionml-runner-{self.model.name}",
                runnable_init_params={"model": self.model},
            ),
        )
        svc = bentoml.Service(self.model.name, runners=[runner])

        @svc.api(
            input=features or infer_feature_io_descriptor(self.model.dataset.feature_type)(),
            output=output or infer_output_io_descriptor(self.model.prediction_type)(),
        )
        def predict(features: typing.Any):
            return runner.predict.run(features)

        return svc

    def serve(
        self,
        model_object: typing.Optional[typing.Any] = None,
        model_version: typing.Optional[str] = None,
        app_version: typing.Optional[str] = None,
        model_file: typing.Optional[typing.Union[str, Path]] = None,
        loader_kwargs: typing.Optional[dict] = None,
    ):
        self.model.artifact = resolve_model_artifact(
            model=self.model,
            model_object=model_object,
            model_version=model_version,
            app_version=app_version,
            model_file=model_file,
            loader_kwargs=loader_kwargs,
        )
        return self.svc


class UnionMLRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu", "nvidia.com/gpu")
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, model: Model):
        self.model: Model = model

    @bentoml.Runnable.method(batchable=False)
    def predict(self, features: typing.Any) -> typing.Any:
        features = self.model.dataset.get_features(features)
        return self.model.predict(features=features)


def infer_feature_io_descriptor(type_: typing.Type) -> typing.Type[bentoml.io.IODescriptor]:
    io_desc = infer_io_descriptor(type_)
    if io_desc is None:
        raise TypeError(
            f"bentoml.io.IODescriptor not found for feature type {type_}. Specify the `features` argument in the "
            "BentoMLServices constructor."
        )
    return io_desc


def infer_output_io_descriptor(type_: typing.Type) -> typing.Type[bentoml.io.IODescriptor]:
    io_desc = infer_io_descriptor(type_)
    if io_desc is None:
        raise TypeError(
            f"bentoml.io.IODescriptor not found for output type {type_}. Specify the `output` argument in the "
            "BentoMLServices constructor."
        )
    return io_desc


def infer_io_descriptor(type_: typing.Type) -> typing.Optional[typing.Type[bentoml.io.IODescriptor]]:
    types_ = get_args(type_)
    if R in types_:
        type_ = types_[-1]

    origin_type_ = get_origin(type_)
    if origin_type_ is not None:
        type_ = origin_type_

    io_desc = {
        np.ndarray: bentoml.io.NumpyNdarray,
        pd.DataFrame: bentoml.io.PandasDataFrame,
        list: bentoml.io.JSON,
        dict: bentoml.io.JSON,
    }.get(type_)

    return io_desc
