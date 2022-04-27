"""Dataset class for defining data source, splitting, parsing, and iteration."""

from inspect import Parameter, signature
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import pandas as pd
from flytekit.core.tracker import TrackedInstance
from flytekit.extras.sqlite3.task import SQLite3Task
from sklearn.model_selection import train_test_split

from unionml.utils import inner_task


class Dataset(TrackedInstance):
    def __init__(
        self,
        name: str = "dataset",
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

        # default component functions
        self._splitter = self._default_splitter
        self._parser = self._default_parser
        self._feature_processor = self._default_feature_processor

        self._reader = None
        self._reader_input_types: Optional[List[Parameter]] = None
        self._reader_return_type: Optional[Dict[str, Type]] = None
        self._labeller = None
        self._dataset_task = None

    def reader(self, fn, **reader_task_kwargs):
        self._reader = fn
        self._reader_task_kwargs = reader_task_kwargs
        return fn

    def splitter(self, fn):
        # TODO: make sure function signature matches expected signature:
        # - splitter(data, test_size, shuffle, random_state)
        self._splitter = fn
        return fn

    def parser(self, fn):
        # TODO: make sure function signature matches expected signature:
        # - parser(data, features, targets)
        self._parser = fn
        return fn

    def feature_processor(self, fn):
        self._feature_processor = fn
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
            unionml_obj=self,
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

    def get_data(self, raw_data):
        train_split, test_split = self._splitter(raw_data, **self.splitter_kwargs)
        # TODO: make this more generic so as to include a validation split
        train_data = self._parser(train_split, **self.parser_kwargs)
        test_data = self._parser(test_split, **self.parser_kwargs)
        return {
            "train": train_data,
            "test": test_data,
        }

    def get_features(self, data):
        data_param, *_ = signature(self._parser).parameters.values()
        data_type = data_param.annotation
        parsed_data = self._parser(data_type(data), self._features, self._targets)
        return self._feature_processor(parsed_data)

    @property
    def reader_input_types(self) -> Optional[List[Parameter]]:
        if self._reader and self._reader_input_types is None:
            return [p for p in signature(self._reader).parameters]
        return self._reader_input_types

    @property
    def reader_return_type(self) -> Dict[str, Type]:
        if self._reader and self._reader_return_type is None:
            return {"data": signature(self._reader).return_annotation}
        elif self._reader_return_type is not None:
            return self._reader_return_type
        raise ValueError(
            "reader_return_type is not defined. Please define a @dataset.reader function with an output annotation."
        )

    @classmethod
    def _from_flytekit_task(
        cls,
        task,
        *args,
        **kwargs,
    ) -> "Dataset":
        dataset = cls(*args, **kwargs)
        dataset._dataset_task = task
        dataset._reader_return_type = task.python_interface.outputs
        dataset._reader_input_types = [
            Parameter(k, Parameter.KEYWORD_ONLY, annotation=v) for k, v in task.python_interface.inputs.items()
        ]
        return dataset

    @classmethod
    def from_sqlite_task(
        cls,
        task: SQLite3Task,
        *args,
        **kwargs,
    ) -> "Dataset":
        return cls._from_flytekit_task(task, *args, **kwargs)

    @classmethod
    def from_sqlalchemy_task(
        cls,
        task: "flytekitplugins.sqlalchemy.SQLAlchemyTask",  # type: ignore
        *args,
        **kwargs,
    ) -> "Dataset":
        return cls._from_flytekit_task(task, *args, **kwargs)

    def _default_splitter(
        self,
        data: pd.DataFrame,
        test_size: float,
        shuffle: bool,
        random_state: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle)

    def _default_parser(
        self, data: pd.DataFrame, features: List[str], targets: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if not features:
            features = [col for col in data if col not in targets]
        try:
            target_data = data[targets]
        except KeyError:
            target_data = pd.DataFrame()
        return data[features], target_data

    def _default_feature_processor(self, data: Tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        return data[0]
