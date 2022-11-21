"""Service definition base class."""

import typing
from pathlib import Path

from unionml.exceptions import ModelArtifactNotFound
from unionml.model import Model

S = typing.TypeVar("S")  # service type


class Service(typing.Generic[S]):
    """UnionML Service base class."""

    def __init__(self, model: Model, name: typing.Optional[str], *args, **kwargs):
        """Initialize a UnionML Service object.

        :param model: the :class:`~unionml.model.Model` bound to this service.
        """
        self._name = name
        self._model = model

    @property
    def model(self) -> Model:
        """Return a :class:`~unionml.model.Model` object."""
        return self._model

    @property
    def name(self) -> str:
        """Return name of the service."""
        return self._name or self.model.name

    def create(self, **kwargs) -> S:
        """Create serving api for particular serving framework.

        :param kwargs: custom keyword arguments to use in the method implementation of a subclass.
        """
        if self.model.artifact is None:
            raise ModelArtifactNotFound(
                f"unionml.Model object '{self.model}' bound to the service '{self}' not found. "
                "Use the BentoMLService.serve() method to specify a model artifact to use for this service."
            )
        return typing.cast(S, None)

    def serve(
        self,
        model_object: typing.Optional[typing.Any] = None,
        model_version: typing.Optional[str] = None,
        app_version: typing.Optional[str] = None,
        model_file: typing.Optional[typing.Union[str, Path]] = None,
        loader_kwargs: typing.Optional[dict] = None,
    ):
        """Serve a specific model object from memory, file, or Flyte cluster.

        This method should assign a :class:`~unionml.model.ModelArtifact` to the bound :class:`~unionml.model.Model`
        provided at initialization.

        If no arguments are provided, this method assumes that the bound :class:`~unionml.model.Model` provided at
        initialization already has a defined :attr:`~unionml.model.Model.artifact`.

        :param model_object: model object to use for prediction.
        :param model_version: model version identifier to use for prediction.
        :param app_version: if ``model_version`` is specified, this argument indicates the app version to use for
            fetching the model artifact.
        :param model_file: a filepath to a serialized model object.
        :param loader_kwargs: additional keyword arguments to be forwarded to the :meth:`unionml.model.Model.loader`
            function.
        """
        self.model.artifact = self.model.resolve_model_artifact(
            model_object=model_object,
            model_version=model_version,
            app_version=app_version,
            model_file=model_file,
            loader_kwargs=loader_kwargs,
        )
