import importlib.util
import inspect
from pathlib import Path
from typing import Optional

from flytekit.core import tracker
from flytekit.exceptions import system
from flytekit.loggers import logger


def import_module_from_file(module_name, file):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise ModuleNotFoundError(f"Module from file {file} cannot be loaded") from exc


class InstanceTrackingMeta(tracker.InstanceTrackingMeta):
    @staticmethod
    def _get_module_from_main(globals) -> Optional[str]:
        curdir = str(Path().absolute())
        file = globals.get("__file__")
        if file is None:
            return None
        if not file.startswith(curdir):
            return None
        module_components = file.replace(f"{curdir}/", "").replace(".py", "").split("/")
        module_name = ".".join(module_components)
        if len(module_components) == 0:
            return None
        return import_module_from_file(module_name, file)

    @staticmethod
    def _find_instance_module():
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == "<module>" and "__name__" in frame.f_globals:
                if frame.f_globals["__name__"] == "__main__":
                    # if the remote_deploy command is invoked in the same module as where
                    # the app is defined,
                    mod = InstanceTrackingMeta._get_module_from_main(frame.f_globals)
                    if mod is None:
                        return None, None
                    return mod.__name__, mod.__file__
                return frame.f_globals["__name__"], frame.f_globals["__file__"]
            frame = frame.f_back
        return None, None

    def __call__(cls, *args, **kwargs):
        o = super(InstanceTrackingMeta, cls).__call__(*args, **kwargs)
        mod_name, mod_file = InstanceTrackingMeta._find_instance_module()
        o._instantiated_in = mod_name
        o._module_file = mod_file
        return o


class TrackedInstance(tracker.TrackedInstance, metaclass=InstanceTrackingMeta):
    """
    Please see the notes for the metaclass above first.

    This functionality has two use-cases currently,
    * Keep track of naming for non-function ``PythonAutoContainerTasks``.  That is, things like the
      :py:class:`flytekit.extras.sqlite3.task.SQLite3Task` task.
    * Task resolvers, because task resolvers are instances of :py:class:`flytekit.core.python_auto_container.TaskResolverMixin`
      classes, not the classes themselves, which means we need to look on the left hand side of them to see how to
      find them at task execution time.
    """

    def __init__(self, *args, **kwargs):
        self._instantiated_in = None
        self._module_file = None
        self._lhs = None
        super().__init__(*args, **kwargs)

    def find_lhs(self) -> str:
        try:
            return super().find_lhs()
        except (system.FlyteSystemException, ModuleNotFoundError):
            module = import_module_from_file(self._instantiated_in, self._module_file)
            for k in dir(module):
                try:
                    candidate = getattr(module, k)
                    # consider the variable equivalent to self if it's of the same type, name
                    if (
                        type(candidate) == type(self)
                        and candidate.__dict__.get("name") == self.__dict__.get("name")
                        and candidate.instantiated_in == self.instantiated_in
                    ):
                        self._lhs = k
                        return k
                except ValueError as err:
                    logger.warning(f"Caught ValueError {err} while attempting to auto-assign name")
                    pass

        logger.error(f"Could not find LHS for {self} in {self._instantiated_in}")
        raise system.FlyteSystemException(f"Error looking for LHS in {self._instantiated_in}")
