import importlib
from typing import List

from flytekit.core.base_task import TaskResolverMixin
from flytekit.core.context_manager import SerializationSettings
from flytekit.core.python_auto_container import PythonAutoContainerTask
from flytekit.core.tracker import TrackedInstance


class TaskResolver(TrackedInstance, TaskResolverMixin):
    """Task Resolver for ulearn"""

    def name(self) -> str:
        return "TaskResolver"

    def load_task(self, loader_args: List[str]) -> PythonAutoContainerTask:
        _, app_module, _, ulearn_obj, _, task_name, *_ = loader_args

        _ulearn_obj = getattr(importlib.import_module(app_module), ulearn_obj)
        task_method = getattr(_ulearn_obj, task_name)
        return task_method()

    def loader_args(self, settings: SerializationSettings, task: PythonAutoContainerTask) -> List[str]:
        return [
            "app-module",
            task.task_function.__ulearn_object__.instantiated_in,
            "ulearn-obj-name",
            task.task_function.__ulearn_object__.lhs,
            "task-name",
            task.task_function.__name__,
        ]


task_resolver = TaskResolver()
