"""Service definition base class."""

import typing
from pathlib import Path

T = typing.TypeVar("T")


class Service(typing.Generic[T]):
    def serve(
        self,
        model_object: typing.Optional[typing.Any] = None,
        model_version: typing.Optional[str] = None,
        app_version: typing.Optional[str] = None,
        model_file: typing.Optional[typing.Union[str, Path]] = None,
        loader_kwargs: typing.Optional[dict] = None,
    ) -> T:
        raise NotImplementedError
