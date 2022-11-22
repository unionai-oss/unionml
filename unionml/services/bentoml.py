"""Service definition for BentoML."""

import asyncio
import typing

try:
    from typing import get_args, get_origin  # type: ignore
except ImportError:
    from typing_extensions import get_args, get_origin

import bentoml
import numpy as np
import pandas as pd

from unionml.dataset import FeatureTypeUnion
from unionml.exceptions import ModelArtifactNotFound
from unionml.model import Model

if typing.TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod

    class RunnerImpl(bentoml.Runner):
        predict: RunnerMethod


class ServiceNotConfigured(Exception):
    """Raised when a service is not configured."""

    ...


class BentoMLService:

    IO_DESCRIPTOR_MAPPING = {
        np.ndarray: bentoml.io.NumpyNdarray,
        pd.DataFrame: bentoml.io.PandasDataFrame,
        list: bentoml.io.JSON,
        dict: bentoml.io.JSON,
    }
    """Maps python types to `BentoML IO descriptors <https://docs.bentoml.org/en/v1.0.10/reference/api_io_descriptors.html>`__"""

    def __init__(self, model: Model, framework: str, name: typing.Optional[str] = None):
        """Initialize a BentoML Service for generating predictions from a :class:`unionml.model.Model`.

        :param model: the :class:`~unionml.model.Model` bound to this service.
        :param framework: machine learning framework supported by :doc:`bentoml <bentoml:reference/frameworks/index>`.
            This is used to access the appropriate module ``bentoml.<framework>``, e.g. ``bentoml.sklearn``.
        :param name: custom name to give this service. If ``None``, uses :attr:`~unionml.model.Model.nam`.
        """
        self._model = model
        self._framework = framework
        self._name = name
        self._bento_model = None
        self._svc = None

    @property
    def model(self) -> Model:
        """Get the :class:`~unionml.model.Model` bound to this service."""
        return self._model

    @property
    def name(self) -> str:
        """Get the name of the service."""
        return self._name or self.model.name

    @property
    def svc(self) -> bentoml.Service:
        """Get the underlying :class:`bentoml.Service` instance."""
        if self._svc is None:
            raise ServiceNotConfigured("Service not configured. Call the `.configure()` method first.")
        return self._svc

    def configure(
        self,
        features: typing.Optional[bentoml.io.IODescriptor] = None,
        output: typing.Optional[bentoml.io.IODescriptor] = None,
        enable_async: bool = False,
        supported_resources: typing.Optional[typing.Tuple] = None,
        supports_cpu_multi_threading: bool = False,
        runnable_method_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> bentoml.Service:
        """Create the bentoml.Service API.

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

        :raises: :class:`~unionml.exceptions.ModelArtifactNotFound` if the bound
            :class:`unionml.model.Model` instance does not have a defined :attr:`~unionml.model.Model.artifact`
            property.
        """
        if self.model.artifact is None:
            raise ModelArtifactNotFound(
                f"unionml.Model object '{self.model}' bound to the service '{self}' not found. "
                "Use the BentoMLService.serve() method to specify a model artifact to use for this service."
            )
        if self._bento_model is None:
            raise ValueError("The _bento_model attribute needs to be set by call the `load_model` method.")

        runner = typing.cast(
            "RunnerImpl",
            bentoml.Runner(
                create_runnable(
                    enable_async,
                    supported_resources,
                    supports_cpu_multi_threading,
                    runnable_method_kwargs,
                ),
                name=f"unionml-runner-{self.name}",
                # this is necessary for `bentoml build` to copy the saved model over to the build directory.
                models=[self._bento_model],
                runnable_init_params={"model": self.model},
            ),
        )
        self._svc = create_service(
            name=self.name,
            runner=runner,
            features=features or infer_feature_io_descriptor(self.model.dataset.feature_type)(),
            output=output or infer_output_io_descriptor(self.model.prediction_type)(),
            enable_async=enable_async,
        )
        return self._svc

    def save_model(self, model_object: typing.Optional[typing.Any] = None, **kwargs):
        """Save the model as a :class:`bentoml.Model`.

        :param model_object: model object to save. If None, this method assumes that
            :class:`unionml.model.Model.artifact` is defined.
        :param framework: machine learning framework supported by :doc:`bentoml <bentoml:reference/frameworks/index>`.
            This is used to access the appropriate module ``bentoml.<framework>``, e.g. ``bentoml.sklearn``
        """
        if model_object is None and self.model.artifact is None:
            raise ModelArtifactNotFound(
                f"unionml.Model object '{self.model}' bound to the service '{self}' not found. "
                "Use the BentoMLService.serve() method to specify a model artifact to use for this service."
            )
        return getattr(bentoml, self._framework).save_model(self.name, model_object, **kwargs)

    def load_model(self, tag_or_version: str):
        """Load a :class:`bentoml.Model` from a version.

        :param tag_or_version: the tag or version associated with the model name. E.g., the version in the tag
            ``my_model:version1`` would be ``version1``.
        :param framework: machine learning framework supported by :doc:`bentoml <bentoml:reference/frameworks/index>`.
            This is used to access the appropriate module ``bentoml.<framework>``, e.g. ``bentoml.sklearn``
        """
        if ":" not in tag_or_version:
            tag = f"{self.name}:{tag_or_version}"
        else:
            tag = tag_or_version
        self._bento_model = getattr(bentoml, self._framework).get(tag)
        model_object = getattr(bentoml, self._framework).load_model(tag)
        self.model.artifact = self.model.resolve_model_artifact(model_object=model_object)


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


def create_runnable(
    enable_async: bool = False,
    supported_resources: typing.Optional[typing.Tuple] = None,
    supports_cpu_multi_threading: bool = False,
    runnable_method_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> bentoml.Runnable:

    _runnable_method_kwargs = {
        "batchable": False,
    }
    _runnable_method_kwargs.update(runnable_method_kwargs or {})

    class UnionMLRunnable(bentoml.Runnable):
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

    return UnionMLRunnable


def infer_feature_io_descriptor(type_: typing.Type) -> typing.Type[bentoml.io.IODescriptor]:
    """Converts a feature input type to a :class:`bentoml.io.IODescriptor`."""
    io_desc = infer_io_descriptor(type_)
    if io_desc is None:
        raise TypeError(
            f"bentoml.io.IODescriptor not found for feature type {type_}. Specify the `features` argument in the "
            "BentoMLServices constructor."
        )
    return io_desc


def infer_output_io_descriptor(type_: typing.Type) -> typing.Type[bentoml.io.IODescriptor]:
    """Converts a prediction output type to a :class:`bentoml.io.IODescriptor`."""
    io_desc = infer_io_descriptor(type_)
    if io_desc is None:
        raise TypeError(
            f"bentoml.io.IODescriptor not found for output type {type_}. Specify the `output` argument in the "
            "BentoMLServices constructor."
        )
    return io_desc


def infer_io_descriptor(type_: typing.Type) -> typing.Optional[typing.Type[bentoml.io.IODescriptor]]:
    """Converts a type to a :class:`bentoml.io.IODescriptor`."""
    origin_type_ = get_origin(type_)

    if origin_type_ is FeatureTypeUnion:
        _, type_ = get_args(type_)
    elif origin_type_ is not None:
        type_ = origin_type_

    return BentoMLService.IO_DESCRIPTOR_MAPPING.get(type_)
