"""Dataset class for defining data source, splitting, parsing, and iteration."""

from functools import partial
from inspect import signature, Parameter
from typing import Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
from flytekit import task, workflow
from flytekit.core.tracker import TrackedInstance
from sklearn.model_selection import train_test_split

from flytekit_learn.task_resolver import task_resolver
from flytekit_learn.utils import inner_task


def dataset_workflow(
    wf_name,
    reader,
    splitter,
    parser,
    reader_kwargs,
    splitter_kwargs,
    parser_kwargs,
):
    def wf():
        # TODO: reader_kwargs needs to be the input of wf()
        data = reader(**reader_kwargs)
        train, test = splitter(data=data, **splitter_kwargs)
        train_data = parser(data=train, **parser_kwargs)
        test_data = parser(data=test, **parser_kwargs)
        return train_data, test_data

    parser_output = signature(parser).return_annotation
    wf.__signature__ = signature(wf).replace(
        return_annotation=NamedTuple("Data", train=parser_output, test=parser_output)
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf


def features_workflow(wf_name, reader, parser, unpack_features, parser_kwargs):

    def wf(features):
        data = parser(data=features, **parser_kwargs)
        return unpack_features(data=data)

    wf.__signature__ = signature(wf).replace(
        parameters=[
            Parameter(
                "features", annotation=signature(reader).return_annotation, kind=Parameter.KEYWORD_ONLY
            ),
        ],
        return_annotation=signature(unpack_features).return_annotation,
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf



def literal_data_workflow(wf_name, data, reader, splitter, parser, splitter_kwargs, parser_kwargs):

    input_type = signature(reader).return_annotation
    if not isinstance(data, input_type):
        data = input_type(data)

    def wf():
        train, test = splitter(data=data, **splitter_kwargs)
        train_data = parser(data=train, **parser_kwargs)
        test_data = parser(data=test, **parser_kwargs)
        return train_data, test_data

    parser_output = signature(parser).return_annotation
    wf.__signature__ = signature(wf).replace(
        return_annotation=NamedTuple("Data", train=parser_output, test=parser_output)
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf


def literal_features_workflow(wf_name, features, reader, parser, unpack_features, parser_kwargs):

    data_type = signature(reader).return_annotation
    if not isinstance(features, data_type):
        features = data_type(features)

    def wf():
        data = parser(data=features, **parser_kwargs)
        return unpack_features(data=data)

    wf.__signature__ = signature(wf).replace(
        return_annotation=signature(unpack_features).return_annotation
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf


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
        **dataset_task_kwargs,
    ):
        super().__init__()
        self.name = name
        self._features = [] if features is None else features
        self._targets = targets
        self._test_size = test_size
        self._shuffle = shuffle
        self._random_state = random_state
        self._dataset_task_kwargs = dataset_task_kwargs

        self._dataset_task = None

    @classmethod
    def _set_default(cls, fn=None, *, name):
        if fn is None:
            return partial(cls._set_default, name=name)

        setattr(cls, name, fn)
        return getattr(cls, name)

    def reader(self, fn, **task_kwargs):
        self._reader = fn
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

        parser_output = signature(self._parser).return_annotation

        @inner_task(
            fklearn_obj=self,
            input_parameters=signature(self._reader).parameters,
            return_annotation=NamedTuple("Dataset", train_data=parser_output, test_data=parser_output),
            **self._dataset_task_kwargs,
        )
        def dataset_task(*args, **kwargs):
            data = self._reader(*args, **kwargs)
            train_split, test_split = self._splitter(data=data, **self.splitter_kwargs)
            train_data = self._parser(train_split, **self.parser_kwargs)
            test_data = self._parser(test_split, **self.parser_kwargs)
            return train_data, test_data

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

    def __call__(self, features_only: bool = False, data=None, features=None, **reader_kwargs):
        splitter_kwargs={
            "test_size": self._test_size,
            "shuffle": self._shuffle,
            "random_state": self._random_state,
        }
        parser_kwargs = {
            "features": self._features,
            "targets": self._targets,
        }

        if data is not None and features is not None:
            raise ValueError("can only supply one of `data` or `features`")

        if data is not None:
            return literal_data_workflow(
                self.literal_data_workflow_name,
                data,
                self._reader,
                self._splitter,
                self._parser,
                splitter_kwargs=splitter_kwargs,
                parser_kwargs=parser_kwargs,
            )

        if features is not None:
            return literal_features_workflow(
                self.literal_features_workflow_name,
                features,
                self._reader,
                self._parser,
                self._unpack_features,
                parser_kwargs=parser_kwargs,
            )

        if features_only:
            return features_workflow(
                self.features_workflow_name,
                self._reader,
                self._parser,
                self._unpack_features,
                parser_kwargs
            )
        
        return dataset_workflow(
            self.data_workflow_name,
            self._reader,
            self._splitter,
            self._parser,
            reader_kwargs=reader_kwargs,
            splitter_kwargs=splitter_kwargs,
            parser_kwargs=parser_kwargs,
        )


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
