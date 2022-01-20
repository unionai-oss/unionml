import importlib
from typing import List

from flytekit.core.base_task import TaskResolverMixin
from flytekit.core.context_manager import SerializationSettings
from flytekit.core.python_auto_container import PythonAutoContainerTask
from flytekit.core.tracker import TrackedInstance


class TaskResolver(TrackedInstance, TaskResolverMixin):
    """Task Resolver for flytekit-learn"""

    def name(self) -> str:
        return "TaskResolver"

    def load_task(self, loader_args: List[str]) -> PythonAutoContainerTask:
        _, app_module, _, fklearn_obj, _, task_name, *_ = loader_args

        _fklearn_obj = getattr(importlib.import_module(app_module), fklearn_obj)
        task_method = getattr(_fklearn_obj, task_name)
        return task_method()

    def loader_args(self, settings: SerializationSettings, task: PythonAutoContainerTask) -> List[str]:
        return [
            "app-module",
            task.task_function.__fklearn_object__.instantiated_in,
            "fklearn-obj-name",
            task.task_function.__fklearn_object__.lhs,
            "task-name",
            task.task_function.__name__,
        ]


task_resolver = TaskResolver()
