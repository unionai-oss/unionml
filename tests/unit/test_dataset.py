import typing
from inspect import signature

import pytest

import pandas as pd
from flytekit.core.python_function_task import PythonFunctionTask

from flytekit_learn import Dataset


@pytest.fixture(scope="function")
def dataset():
    return Dataset(
        targets=["target"],
        test_size=0.2,
        shuffle=True,
        random_state=123,
    )


@pytest.fixture(scope="function")
def reader():
    def _reader(value: float, n_samples: int) -> typing.List[float]:
        return [value for _ in range(n_samples)]
    return _reader


def test_dataset_reader(dataset, reader):

    dataset.reader(reader)
    assert dataset._reader == reader
    assert reader(1, 10) == dataset._reader(1, 10)
    assert reader(2, 10) != dataset._reader(3, 10)


def test_dataset_task(dataset, reader):
    dataset.reader(reader)
    dataset_task = dataset.dataset_task()
    parser_ret_type = signature(dataset._parser).return_annotation
    assert isinstance(dataset_task, PythonFunctionTask)
    assert dataset_task.python_interface.inputs == {"value": float, "n_samples": int}
    assert dataset_task.python_interface.outputs["train_data"] == parser_ret_type
    assert dataset_task.python_interface.outputs["test_data"] == parser_ret_type


def test_dataset_get_features():
    dataset = Dataset(
        features=["x"],
        targets=["y"],
        test_size=0.2,
        shuffle=True,
        random_state=123,
    )
    features = dataset.get_features(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    assert features.equals(pd.DataFrame({"x": [1, 2, 3]}))
