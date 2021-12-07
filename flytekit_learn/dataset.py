"""Dataset class for defining data source, splitting, parsing, and iteration."""

from functools import partial
from inspect import signature, Parameter
from typing import Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
from flytekit import task, workflow
from sklearn.model_selection import train_test_split


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

    parser_output = signature(parser.task_function).return_annotation
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
                "features", annotation=signature(reader.task_function).return_annotation, kind=Parameter.KEYWORD_ONLY
            ),
        ],
        return_annotation=signature(unpack_features.task_function).return_annotation,
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf



def literal_data_workflow(wf_name, data, reader, splitter, parser, splitter_kwargs, parser_kwargs):

    input_type = signature(reader.task_function).return_annotation
    if not isinstance(data, input_type):
        data = input_type(data)

    def wf():
        train, test = splitter(data=data, **splitter_kwargs)
        train_data = parser(data=train, **parser_kwargs)
        test_data = parser(data=test, **parser_kwargs)
        return train_data, test_data

    parser_output = signature(parser.task_function).return_annotation
    wf.__signature__ = signature(wf).replace(
        return_annotation=NamedTuple("Data", train=parser_output, test=parser_output)
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf


def literal_features_workflow(wf_name, features, reader, parser, unpack_features, parser_kwargs):

    data_type = signature(reader.task_function).return_annotation
    if not isinstance(features, data_type):
        features = data_type(features)

    def wf():
        data = parser(data=features, **parser_kwargs)
        return unpack_features(data=data)

    wf.__signature__ = signature(wf).replace(
        return_annotation=signature(unpack_features.task_function).return_annotation
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf


class Dataset:
    
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
        self.name = name
        self._features = [] if features is None else features
        self._targets = targets
        self._test_size = test_size
        self._shuffle = shuffle
        self._random_state = random_state

    @classmethod
    def _set_default(cls, fn=None, *, name):
        if fn is None:
            return partial(cls._set_default, name=name)

        setattr(cls, name, task(fn))
        return getattr(cls, name)

    def reader(self, fn):
        self._reader = task(fn)
        return self._reader

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
    data: pd.DataFrame, test_size: float, shuffle: bool, random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle)


@Dataset._set_default(name="_parser")
def _default_parser(data: pd.DataFrame, features: List[str], targets: List[str]) -> List[pd.DataFrame]:
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
def _default_unpack_features(data: List[pd.DataFrame]) -> pd.DataFrame:
    return data[0]
