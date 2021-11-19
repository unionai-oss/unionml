"""Dataset class for defining data source, splitting, parsing, and iteration."""

from inspect import signature
from typing import List, NamedTuple, Optional, Tuple

import pandas as pd
from flytekit import task, workflow
from sklearn.model_selection import train_test_split


def _default_splitter(
    data: pd.DataFrame, test_size: float, shuffle: bool, random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        data, test_size=test_size, random_state=random_state, shuffle=shuffle,
    )


def _default_parser(data: pd.DataFrame, features: List[str], targets: List[str]) -> List[pd.DataFrame]:
    return [data[features], data[targets]]


def dataset_workflow(
    reader,
    splitter,
    parser,
    reader_kwargs,
    splitter_kwargs,
    parser_kwargs,
):
    def wf():
        data = reader(**reader_kwargs)
        train, test = splitter(data=data, **splitter_kwargs)
        train_data = parser(data=train, **parser_kwargs)
        test_data = parser(data=test, **parser_kwargs)
        return train_data, test_data

    parser_output = signature(parser.task_function).return_annotation
    wf.__signature__ = signature(wf).replace(
        return_annotation=NamedTuple("Data", train=parser_output, test=parser_output)
    )
    return workflow(wf)


class Dataset:
    
    def __init__(
        self,
        features: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        test_size: float = True,
        shuffle: bool = True,
        random_state: int = 12345,
    ):
        self._features = features
        self._targets = targets
        self._test_size = test_size
        self._shuffle = shuffle
        self._random_state = random_state

        # default functions
        self._splitter = task(_default_splitter)
        self._parser = task(_default_parser)

    def reader(self, fn):
        self._reader = task(fn)
        return fn

    def read(self, **reader_kwargs):
        return dataset_workflow(
            self._reader,
            self._splitter,
            self._parser,
            reader_kwargs=reader_kwargs,
            splitter_kwargs={
                "test_size": self._test_size,
                "shuffle": self._shuffle,
                "random_state": self._random_state,
            },
            parser_kwargs={
                "features": self._features,
                "targets": self._targets,
            }
        )
