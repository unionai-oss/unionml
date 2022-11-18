"""Service definition base class."""

import typing

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
