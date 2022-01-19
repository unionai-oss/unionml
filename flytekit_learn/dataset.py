"""Dataset class for defining data source, splitting, parsing, and iteration."""

from functools import partial
from inspect import signature, Parameter
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type

import pandas as pd
from flytekit.core.tracker import TrackedInstance
from flytekit.extras.sqlite3.task import SQLite3Task
from sklearn.model_selection import train_test_split

from flytekit_learn.utils import inner_task


class Dataset(TrackedInstance):
    
    def __init__(
        self,
        name: str = None,
        *,
        features: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 12345,
    ):
        super().__init__()
        self.name = name
        self._features = [] if features is None else features
        self._targets = targets
        self._test_size = test_size
        self._shuffle = shuffle
        self._random_state = random_state

        self._reader = None
        self._reader_input_types = None
        self._reader_return_type = None
        self._labeller = None
        self._dataset_task = None

    @classmethod
    def _set_default(cls, fn=None, *, name):
        if fn is None:
            return partial(cls._set_default, name=name)

        setattr(cls, name, fn)
        return getattr(cls, name)

    def reader(self, fn, **reader_task_kwargs):
        self._reader = fn
        self._reader_task_kwargs = reader_task_kwargs
        return fn

    @property
    def splitter_kwargs(self):
        return {
            "test_size": self._test_size,
            "shuffle": self._shuffle,
            "random_state": self._random_state,
        }

    @property
    def parser_kwargs(self):
        return {
            "features": self._features,
            "targets": self._targets,
        }

    def dataset_task(self):
        if self._dataset_task:
            return self._dataset_task

        reader_sig = signature(self._reader)

        # TODO: make sure return type is not None
        @inner_task(
            fklearn_obj=self,
            input_parameters=reader_sig.parameters,
            return_annotation=NamedTuple("ReaderOutput", data=reader_sig.return_annotation),
            **self._reader_task_kwargs,
        )
        def dataset_task(*args, **kwargs):
            return self._reader(*args, **kwargs)

        self._dataset_task = dataset_task
        return dataset_task

    @property
    def literal_data_workflow_name(self):
        return f"{self.name}.literal_data"

    @property
    def literal_features_workflow_name(self):
        return f"{self.name}.literal_features"

    @property
    def data_workflow_name(self):
        return f"{self.name}.data"

    @property
    def features_workflow_name(self):
        return f"{self.name}.features"

    def get_features(self, data):
        parsed_data = self._parser(data, self._features, self._targets)
        return self._unpack_features(parsed_data)

    @property
    def reader_input_types(self) -> List[Parameter]:
        if self._reader_input_types is None:
            return list(signature(self._reader).parameters)
        return self._reader_input_types

    @property
    def reader_return_type(self) -> Dict[str, Type]:
        if self._reader_return_type is None:
            return {"data": signature(self._reader).return_annotation}
        return self._reader_return_type

    @classmethod
    def _from_flytekit_task(
        cls: "Dataset",
        task,
        *args,
        **kwargs,
    ) -> "Dataset":
        dataset = cls(*args, **kwargs)
        dataset._dataset_task = task
        dataset._reader_return_type = task.python_interface.outputs
        dataset._reader_input_types = [
            Parameter(k, Parameter.KEYWORD_ONLY, annotation=v)
            for k, v in task.python_interface.inputs.items()
        ]
        return dataset

    @classmethod
    def from_sqlite_task(
        cls: "Dataset",
        task: SQLite3Task,
        *args,
        **kwargs,
    ) -> "Dataset":
        return cls._from_flytekit_task(task, *args, **kwargs)

    @classmethod
    def from_sqlalchemy_task(
        cls: "Dataset",
        task: "flytekitplugins.sqlalchemy.SQLAlchemyTask",  # type: ignore
        *args,
        **kwargs,
    ) -> "Dataset":
        return cls._from_flytekit_task(task, *args, **kwargs)


@Dataset._set_default(name="_splitter")
def _default_splitter(
    self, data: pd.DataFrame, test_size: float, shuffle: bool, random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle)


@Dataset._set_default(name="_parser")
def _default_parser(self, data: pd.DataFrame, features: List[str], targets: List[str]) -> List[pd.DataFrame]:
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if not features:
        features = [col for col in data if col not in targets]
    try:
        targets = data[targets]
    except KeyError:
        targets = pd.DataFrame()
    return [data[features], targets]


@Dataset._set_default(name="_unpack_features")
def _default_unpack_features(self, data: List[pd.DataFrame]) -> pd.DataFrame:
    return data[0]
