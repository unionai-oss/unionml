"""UnionML Services Module."""

from unionml.utils import module_is_installed

from . import base

if module_is_installed("bentoml"):
    from . import bentoml
