import importlib
from typing import TYPE_CHECKING, List

from flytekit.core.base_task import TaskResolverMixin
from flytekit.core.context_manager import SerializationSettings
from flytekit.core.python_auto_container import PythonAutoContainerTask
from flytekit.core.tracker import TrackedInstance

if TYPE_CHECKING:
    from unionml.model import Model
else:

    class Model:
        pass


class TaskResolver(TrackedInstance, TaskResolverMixin):
    """Task Resolver for unionml"""

    def name(self) -> str:
        return "TaskResolver"

    def load_task(self, loader_args: List[str]) -> PythonAutoContainerTask:
        # TODO(review,zevisert): I think this is necessary to make the task resolver work with existing
        # tasks from before the variant was introduced, otherwise we risk breaking existing workflows with
        # a ValueError(not enough values to unpack). We should remove this once we're confident that all tasks
        # have been updated to provide the variant arg.
        if "task-variant" not in loader_args:
            loader_args.extend(["task-variant", ""])

        _, app_module, _, unionml_obj, _, task_name, _, task_variant, *_ = loader_args

        _unionml_obj = getattr(importlib.import_module(app_module), unionml_obj)
        task_method = getattr(_unionml_obj, task_name)
        if task_name.startswith("callback"):
            model: Model = _unionml_obj
            return task_method(model.__callbacks[task_variant])

        return task_method()

    def loader_args(self, settings: SerializationSettings, task: PythonAutoContainerTask) -> List[str]:
        return [
            "app-module",
            task.task_function.__unionml_object__.instantiated_in,
            "unionml-obj-name",
            task.task_function.__unionml_object__.lhs,
            "task-name",
            task.task_function.__name__.split("@")[0],
            "task-variant",
            (task.task_config or {}).get("unionml:variant", ""),
        ]


task_resolver = TaskResolver()
