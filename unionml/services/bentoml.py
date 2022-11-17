"""Service definition for BentoML."""

import asyncio
import typing

try:
    from typing import get_origin  # type: ignore
except ImportError:
    from typing_extensions import get_origin

import bentoml
import numpy as np
import pandas as pd

from unionml.model import Model
from unionml.services.base import Service

if typing.TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod

    class RunnerImpl(bentoml.Runner):
        predict: RunnerMethod


class BentoMLService(Service[bentoml.Service]):

    IO_DESCRIPTOR_MAPPING = {
        np.ndarray: bentoml.io.NumpyNdarray,
        pd.DataFrame: bentoml.io.PandasDataFrame,
        list: bentoml.io.JSON,
        dict: bentoml.io.JSON,
    }
    """Maps python types to `BentoML IO descriptors <https://docs.bentoml.org/en/v1.0.10/reference/api_io_descriptors.html>`__"""

    def __init__(
        self,
        model: Model,
        features: typing.Optional[bentoml.io.IODescriptor] = None,
        output: typing.Optional[bentoml.io.IODescriptor] = None,
        enable_async: bool = False,
        supported_resources: typing.Optional[typing.Tuple] = None,
        supports_cpu_multi_threading: bool = False,
        runnable_method_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        """Initialize a BentoML Service for generating predictions from a :class:`unionml.model.Model`.

        :param model: the :class:`~unionml.model.Model` bound to this service.
        :param features: `BentoML IO descriptor <https://docs.bentoml.org/en/latest/reference/api_io_descriptors.html>`__
            for the feature data. The descriptor is inferred using the
            :attr:`~unionml.services.bentoml.BentoMLService.IO_DESCRIPTOR_MAPPING` attribute. This must be provided for
            unsupported types.
        :param output: `BentoML IO descriptor <https://docs.bentoml.org/en/latest/reference/api_io_descriptors.html>`__
            for the prediction output. The descriptor is inferred using the
            :attr:`~unionml.services.bentoml.BentoMLService.IO_DESCRIPTOR_MAPPING` attribute. This must be provided for
            unsupported types.
        :param supported_resources: Indicates which resources the ``bentoml.Runnable`` class supports, see
            `here <https://docs.bentoml.org/en/latest/concepts/runner.html#custom-runner>`__ for more details.
        :param supports_cpu_multi_threading: Indicates whether the ``bentoml.Runnable`` class supports multi-threading,
            see `here <https://docs.bentoml.org/en/latest/concepts/runner.html#custom-runner>`__ for more details.
        :param runnable_method_kwargs: Keyword arguments forwarded to
            `bentoml.Runnable.method <https://docs.bentoml.org/en/latest/reference/core.html#bentoml.Runnable.method>`__.
        """
        super().__init__(model)
        self._features = features
        self._output = output
        self._supported_resources = supported_resources
        self._supports_cpu_multi_threading = supports_cpu_multi_threading
        self._runnable_method_kwargs = runnable_method_kwargs
        self._enable_async = enable_async
        self._svc = None

    @property
    def svc(self) -> bentoml.Service:
        """Access the bentoml.Service as a property."""
        if self.model.artifact is None:
            raise ValueError(
                "Model artifact not defined. Invoke the `.serve()` method to specify the model you want to serve."
            )
        if self._svc is not None:
            return self._svc
        self._svc = self.create()
        return self._svc

    def create(self, **kwargs) -> bentoml.Service:
        """Create the bentoml.Service API.

        :raises: :class:`~unionml.exceptions.ModelArtifactNotFound` if the bound
            :class:`unionml.model.Model` instance does not have a defined :attr:`~unionml.model.Model.artifact`
            property.
        """
        super().create()

        runner = typing.cast(
            "RunnerImpl",
            bentoml.Runner(
                create_runner(
                    self._enable_async,
                    self._supported_resources,
                    self._supports_cpu_multi_threading,
                    self._runnable_method_kwargs,
                ),
                name=f"unionml-runner-{self.model.name}",
                runnable_init_params={"model": self.model},
            ),
        )
        service = create_service(
            name=f"unionml_{self.model.name}",
            runner=runner,
            features=self._features or infer_feature_io_descriptor(self.model.dataset.feature_type)(),
            output=self._output or infer_output_io_descriptor(self.model.prediction_type)(),
            enable_async=self._enable_async,
        )
        if self._svc is None:
            self._svc = service
        return service


def create_service(
    name: str,
    runner: bentoml.Runnable,
    features: typing.Optional[bentoml.io.IODescriptor] = None,
    output: typing.Optional[bentoml.io.IODescriptor] = None,
    enable_async: bool = False,
) -> bentoml.Service:
    """Create :class:`bentoml.Service`."""
    svc = bentoml.Service(name, runners=[runner])

    if enable_async:

        @svc.api(input=features, output=output)
        async def predict(features: typing.Any):
            result = await runner.predict.async_run(features)
            return await asyncio.gather(result)

    else:

        @svc.api(input=features, output=output)
        def predict(features: typing.Any):
            return runner.predict.run(features)

    return svc


def create_runner(
    enable_async: bool = False,
    supported_resources: typing.Optional[typing.Tuple] = None,
    supports_cpu_multi_threading: bool = False,
    runnable_method_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> bentoml.Runnable:

    _runnable_method_kwargs = {
        "batchable": False,
    }
    _runnable_method_kwargs.update(runnable_method_kwargs or {})

    class UnionMLRunner(bentoml.Runnable):
        SUPPORTED_RESOURCES = supported_resources or ("cpu", "nvidia.com/gpu")
        SUPPORTS_CPU_MULTI_THREADING = supports_cpu_multi_threading

        def __init__(self, model: Model):
            self.model: Model = model

        if enable_async:

            @bentoml.Runnable.method(**_runnable_method_kwargs)
            async def predict(self, features: typing.Any) -> typing.Any:
                features = self.model.dataset.get_features(features)
                return self.model.predict(features=features)

        else:

            @bentoml.Runnable.method(**_runnable_method_kwargs)
            def predict(self, features: typing.Any) -> typing.Any:
                features = self.model.dataset.get_features(features)
                return self.model.predict(features=features)

    return UnionMLRunner


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
    origin_type_ = get_origin(type_)
    if origin_type_ is not None:
        type_ = origin_type_
    return BentoMLService.IO_DESCRIPTOR_MAPPING.get(type_)
