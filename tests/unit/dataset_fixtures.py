import json
import typing

import pytest

from unionml import Dataset

N_SAMPLES = 100
TEST_SIZE = 0.2


@pytest.fixture(scope="function")
def dataset():
    return Dataset(
        features=["x"],
        targets=["y"],
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=123,
    )


@pytest.fixture(scope="function")
def simple_reader():
    def _reader(value: float, n_samples: int) -> typing.List[float]:
        return [value for _ in range(n_samples)]

    return _reader


@pytest.fixture(scope="function")
def dict_dataset_reader():
    def _reader() -> typing.List[typing.Dict[str, int]]:
        return [{"x": i, "y": i * 2} for i in range(1, N_SAMPLES + 1)]

    return _reader


@pytest.fixture(scope="function")
def reader_with_json_string_output(dict_dataset_reader):
    def _reader() -> str:
        return json.dumps(dict_dataset_reader())

    return _reader
