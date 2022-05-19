"""Dataset class for defining data source, splitting, parsing, and iteration."""

import json
from functools import partial
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar, cast

import pandas as pd
from flytekit.core.tracker import TrackedInstance
from flytekit.extras.sqlite3.task import SQLite3Task
from sklearn.model_selection import train_test_split

from unionml.utils import inner_task

R = TypeVar("R")  # raw data
D = TypeVar("D")  # model-ready data


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
        self._loader = self._default_loader
        self._splitter = self._default_splitter
        self._parser = self._default_parser
        self._parser_feature_key: int = 0  # assume that first element of parser tuple output contains features
        self._feature_loader = self._default_feature_loader
        self._feature_transformer = self._default_feature_transformer

        self._reader = None
        self._reader_input_types: Optional[List[Parameter]] = None
        self._reader_return_type: Optional[Dict[str, Type]] = None
        self._dataset_task = None

    def reader(self, fn=None, **reader_task_kwargs):
        """Register a reader function for getting data from some external source."""
        if fn is None:
            return partial(self.reader, **reader_task_kwargs)
        self._reader = fn
        self._reader_task_kwargs = reader_task_kwargs
        return fn

    def loader(self, fn):
        """Register an optional loader function for loading data into memory for model training.

        This function should take the output of the reader function and return the data structure needed
        for model training.
        """
        self._loader = fn
        return fn

    def splitter(self, fn):
        """Register an optional splitter function that partitions data into training and test sets."""
        self._splitter = fn
        return fn

    def parser(self, fn, feature_key: int = 0):
        """Register an optional parser function that produces a tuple of features and targets."""
        self._parser = fn
        self._parser_feature_key = feature_key
        return fn

    def feature_loader(self, fn):
        """Register an optional feature loader that loads data from some serialized format into raw features.

        This function handles prediction cases in two contexts:
        1. When the `unionml predict` cli command is invoked with the --features flag.
        2. When the FastAPI app `/predict/` endpoint is invoked with features passed in as JSON or string encoding.

        And it should return the data structure needed for model training.
        """
        self._feature_loader = fn
        return fn

    def feature_transformer(self, fn):
        """Register an optional feature transformer that performs pre-processing on features before prediction.

        This function handles prediction cases in three contexts:
        1. When the `unionml predict` cli command is invoked with the --features flag.
        2. When the FastAPI app `/predict/` endpoint is invoked with features passed in as JSON or string encoding.
        3. When the `model.predict` or `model.remote_predict` functions are invoked.
        ."""
        self._feature_transformer = fn
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
        data = self._loader(raw_data)
        splits = self._splitter(data, **self.splitter_kwargs)
        if len(splits) == 1:
            return {"train": self._parser(splits[0], **self.parser_kwargs)}

        train_split, test_split = splits
        # TODO: make this more generic so as to include a validation split
        train_data = self._parser(train_split, **self.parser_kwargs)
        test_data = self._parser(test_split, **self.parser_kwargs)
        return {
            "train": train_data,
            "test": test_data,
        }

    def get_features(self, features):
        parsed_data = self._parser(self._feature_loader(features), self._features, self._targets)
        features = parsed_data[self._parser_feature_key]
        return self._feature_transformer(features)

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

    def _default_loader(self, data: R) -> R:
        [(_, data_type)] = self.reader_return_type.items()
        if data_type is pd.DataFrame:
            return pd.DataFrame(data)
        return data

    def _default_splitter(
        self,
        data: D,
        test_size: float,
        shuffle: bool,
        random_state: int,
    ) -> Tuple[D, ...]:
        if not isinstance(data, pd.DataFrame):
            return (data,)
        return train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle)

    def _default_parser(self, data: D, features: Optional[List[str]], targets: Optional[List[str]]) -> Tuple[D, ...]:
        if not isinstance(data, pd.DataFrame):
            return (data,)

        if features is not None and targets is not None:
            features = [col for col in data if col not in targets]
        try:
            target_data = data[targets]
        except KeyError:
            target_data = pd.DataFrame()
        return data[features], target_data

    def _default_feature_loader(self, features: Any) -> R:
        if isinstance(features, Path):
            with features.open() as f:
                features = json.load(f)

        [(_, data_type)] = self.reader_return_type.items()
        if data_type is pd.DataFrame:
            return pd.DataFrame(features)
        return features

    def _default_feature_transformer(self, features: R) -> D:
        """
        By default this is just a pass-through function. The user can choose to override it with
        the `@dataset.feature_transformer` decorator.
        """
        return cast(D, features)
