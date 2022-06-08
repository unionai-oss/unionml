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
        """Initialize a UnionML Dataset.

        The term *UnionML Dataset* refers to the specification of data used to train a model object from features
        and targets (see :py:class:`unionml.model.Model` for more details) or generate predictions from one
        based on some features. This specification is implemented by the used via the functional entrypoints,
        e.g. :meth:`unionml.dataset.Dataset.reader`.

        By default the *UnionML Dataset* knows how to handle ``pandas.DataFrame`` objects automatically, meaning that
        the only function that needs to be implemented is the :meth:`unionml.dataset.Dataset.reader`. To add support
        for other data structures, the user needs to implement the rest of the functional entrypoints.

        :param name: name of the dataset.
        :param features: a list of string keys used to access features from the data structure. The type of this data
            structure is determined by the output of the :meth:`unionml.dataset.Dataset.reader` by default, but
            if :meth:`unionml.dataset.Dataset.loader` is implemented then the output type of the latter function
            is taken.
        :param targets: a list of string keys used to access targets data. The type of this data is determined in the
            same way as the ``features`` argument.
        :param test_size: the percent of the dataset to split out as the test set.
        :param shuffle: if True, shuffles the dataset before dataset splitting.
        :param random_state: random state used for data shuffling.
        """
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
        """Register a reader function for getting data from some external source.

        The signature of this function is flexible and dependent on the use case.

        :param fn: function to register
        """
        if fn is None:
            return partial(self.reader, **reader_task_kwargs)
        self._reader = fn
        self._reader_task_kwargs = reader_task_kwargs
        return fn

    def loader(self, fn):
        """Register an optional loader function for loading data into memory for model training.

        This function should take the output of the reader function and return the data structure needed
        for model training. If specified, the output type of this function take precedence over that of the ``reader``
        function and the type signatures of ``splitter`` and ``parser`` should adhere to it.

        By default this is simply a pass through function that returns the output of the ``reader`` function.

        :param fn: function to register
        """
        self._loader = fn
        return fn

    def splitter(self, fn):
        """Register an optional splitter function that partitions data into training and test sets.

        :param fn: function to register

        The following is equivalent to the default implementation.

        .. code-block:: python

            from typing import Tuple

            Splits = Tuple[pd.DataFrame, pd.DataFrame]

            @dataset.splitter
            def splitter(data: pd.DataFrame, test_size: float, shuffle: bool, random_state: int) -> Splits:
                if shuffle:
                    data = data.sample(frac=1.0, random_state=random_state)
                n = int(data.shape[0] * test_size)
                return data.iloc[:-n], data.iloc[-n:]
        """
        self._splitter = fn
        return fn

    def parser(self, fn, feature_key: int = 0):
        """Register an optional parser function that produces a tuple of features and targets.

        :param fn: function to register
        :param feature_key: the index of the features in the output of the parser function. By default, this assumes
            that the first element of the output contains the features.

        The following is equivalent to the default implementation.

        .. code-block:: python

            from typing import Optional, Tuple

            Parsed = Tuple[pd.DataFrame, pd.DataFrame]

            @dataset.parser
            def parser(data: pd.DataFrame, features: Optional[List[str]], targets: List[str]) -> Parsed:
                if not features:
                    features = [col for col in data if col not in targets]
                return data[features], data[targets]
        """
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
        """
        self._feature_transformer = fn
        return fn

    @property
    def splitter_kwargs(self):
        """The keyword arguments to be forwarded to the splitter function."""
        return {
            "test_size": self._test_size,
            "shuffle": self._shuffle,
            "random_state": self._random_state,
        }

    @property
    def parser_kwargs(self):
        """The keyword arguments to be forwarded to the parser function."""
        return {
            "features": self._features,
            "targets": self._targets,
        }

    def dataset_task(self):
        """Create a Flyte task for getting the dataset using the ``reader`` function."""
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

    def get_data(self, raw_data):
        """Get training data from from its raw form to its model-ready form.

        :param raw_data: Raw data in the same form as the ``reader`` output.

        This function uses the following registered functions to create parsed, split data:

        - :meth:`unionml.dataset.Dataset.loader`
        - :meth:`unionml.dataset.Dataset.splitter`
        - :meth:`unionml.dataset.Dataset.parser`
        """
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
        """Get feature data from its raw form to its model-ready form.

        This function uses the following registered functions to create model-ready features:

        - :meth:`unionml.dataset.Dataset.feature_loader`
        - :meth:`unionml.dataset.Dataset.parser`
        - :meth:`unionml.dataset.Dataset.feature_transformer`
        """
        parsed_data = self._parser(self._feature_loader(features), self._features, self._targets)
        features = parsed_data[self._parser_feature_key]
        return self._feature_transformer(features)

    @property
    def reader_input_types(self) -> Optional[List[Parameter]]:
        """Get the input parameters of the reader."""
        if self._reader and self._reader_input_types is None:
            return [*signature(self._reader).parameters.values()]
        return self._reader_input_types

    @property
    def reader_return_type(self) -> Dict[str, Type]:
        """Get the output type of the ``reader`` .

        If the ``loader`` is user-defined then the output of that function is used.
        """
        if self._loader != self._default_loader:
            return {"data": signature(self._loader).return_annotation}
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
        """Converts a sqlite task to a dataset.

        This class method creates a *UnionML Dataset** that uses the
        `sqlite task <https://docs.flyte.org/projects/cookbook/en/latest/auto/integrations/flytekit_plugins/sql/sqlite3_integration.html#sphx-glr-auto-integrations-flytekit-plugins-sql-sqlite3-integration-py>`__
        as its ``reader`` function.
        """
        return cls._from_flytekit_task(task, *args, **kwargs)

    @classmethod
    def from_sqlalchemy_task(
        cls,
        task: "flytekitplugins.sqlalchemy.SQLAlchemyTask",  # type: ignore
        *args,
        **kwargs,
    ) -> "Dataset":
        """Converts a sqlalchemy task to a dataset.

        This class method creates a *UnionML Dataset* that uses the
        `sqlalchemy task <https://docs.flyte.org/projects/cookbook/en/latest/auto/integrations/flytekit_plugins/sql/sql_alchemy.html>`__
        as its ``reader`` function.
        """
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
