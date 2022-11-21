"""UnionML Services Module."""

from unionml.utils import module_is_installed

if module_is_installed("bentoml"):
    from . import bentoml
