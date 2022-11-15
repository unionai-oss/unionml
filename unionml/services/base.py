"""Service definition protocol."""

import typing
from pathlib import Path


class Service:
    def serve(
        self,
        model_object: typing.Optional[typing.Any] = None,
        model_version: typing.Optional[str] = None,
        app_version: typing.Optional[str] = None,
        model_file: typing.Optional[typing.Union[str, Path]] = None,
        loader_kwargs: typing.Optional[dict] = None,
    ):
        ...
