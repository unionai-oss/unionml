import random
import typing
from inspect import signature

import pytest

import pandas as pd
from flytekit.core.python_function_task import PythonFunctionTask

from flytekit_learn import Dataset


@pytest.fixture(scope="function")
def dataset():
    return Dataset(
        features=["x"],
        targets=["y"],
        test_size=0.2,
        shuffle=True,
        random_state=123,
    )


@pytest.fixture(scope="function")
def simple_reader():
    def _reader(value: float, n_samples: int) -> typing.List[float]:
        return [value for _ in range(n_samples)]
    return _reader


N_SAMPLES = 100


@pytest.fixture(scope="function")
def dict_dataset_reader():
    def _reader() -> typing.List[typing.Dict[str, int]]:
        return [{"x": i, "y": i * 2} for i in range(1, N_SAMPLES + 1)]
    return _reader


def test_dataset_reader(dataset, simple_reader):
    dataset.reader(simple_reader)
    assert dataset._reader == simple_reader
    assert simple_reader(1, 10) == dataset._reader(1, 10)
    assert simple_reader(2, 10) != dataset._reader(3, 10)


def test_dataset_task(dataset, simple_reader):
    dataset.reader(simple_reader)
    dataset_task = dataset.dataset_task()
    reader_ret_type = signature(dataset._reader).return_annotation
    assert isinstance(dataset_task, PythonFunctionTask)
    assert dataset_task.python_interface.inputs == {"value": float, "n_samples": int}
    assert dataset_task.python_interface.outputs["data"] == reader_ret_type


def test_dataset_get_features(dataset):
    features = dataset.get_features(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    assert features.equals(pd.DataFrame({"x": [1, 2, 3]}))


def test_dataset_custom_splitter_parser(dataset, dict_dataset_reader):
    dataset.reader(dict_dataset_reader)

    Data = typing.List[typing.Dict[str, int]]

    @dataset.splitter
    def splitter(
        data: typing.List[typing.Dict[str, int]],
        test_size: float,
        shuffle: bool,
        random_state: int,
    ) -> typing.Tuple[Data, Data]:
        n = len(data)
        n_test_samples = int(n * test_size)
        random.seed(random_state)

        if shuffle:
            random.shuffle(data)

        return data[:-n_test_samples], data[-n_test_samples:]

    @dataset.parser
    def parser(data, features: typing.List[str], targets: typing.List[str]) -> typing.Tuple[Data, Data]:
        feature_data, target_data = [], []
        for example in data:
            feature_data.append({x: example[x] for x in features})
            target_data.append({x: example[x] for x in targets})
        return feature_data, target_data

    dataset_task = dataset.dataset_task()
    raw_data = dataset_task.task_function()
    data = dataset.get_data(raw_data)

    train_features, train_targets = data["train"]
    test_features, test_targets = data["test"]

    expected_test_size = int(N_SAMPLES * 0.2)

    for train_split in (train_features, train_targets):
        assert len(train_split) == N_SAMPLES - expected_test_size

    for test_split in (test_features, test_targets):
        assert len(test_split) == expected_test_size

    for _features in (train_features, test_features):
        assert all("x" in example for example in _features)
        assert all("y" not in example for example in _features)

    for _targets in (train_targets, test_targets):
        assert all("x" not in example for example in _targets)
        assert all("y" in example for example in _targets)
