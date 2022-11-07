"""Unit tests for dataset and model type guards."""

import typing

import pytest

import unionml.type_guards as type_guards

# In reality this would be a collection-like object like list, dict, dataframe, etc.
EXPECTED_DATA_TYPE = int


def test_guard_reader():
    def valid_reader() -> int:
        ...

    def invalid_reader():
        ...

    type_guards.guard_reader(valid_reader)
    with pytest.raises(TypeError):
        type_guards.guard_reader(invalid_reader)


def test_guard_loader():

    expected_data_type = int

    def valid_loader(data: int):
        ...

    def invalid_loader(data: str):
        ...

    type_guards.guard_loader(valid_loader, expected_data_type)
    with pytest.raises(TypeError):
        type_guards.guard_loader(invalid_loader, expected_data_type)


# splitter types
TupleOutput = typing.Tuple[int, int]
NamedTupleOutput = typing.NamedTuple("Splits", train=int, test=int)  # type: ignore
ListOutput = typing.Tuple[int, int]


# valid splitters
def splitter_tuple(data: int, test_size: float, shuffle: bool, random_state: int) -> TupleOutput:
    ...


def splitter_namedtuple(data: int, test_size: float, shuffle: bool, random_state: int) -> NamedTupleOutput:
    ...


def splitter_list(data: int, test_size: float, shuffle: bool, random_state: int) -> ListOutput:
    ...


# invalid splitters
def splitter_no_test_size(data: int, shuffle: bool, random_state: int) -> TupleOutput:
    ...


def splitter_no_shuffle(data: int, test_size: float, random_state: int) -> TupleOutput:
    ...


def splitter_no_random_state(data: int, test_size: bool, shuffle: bool) -> TupleOutput:
    ...


def splitter_wrong_test_size(data: int, test_size: str, shuffle: bool, random_state: int) -> TupleOutput:
    ...


def splitter_wrong_shuffle(data: int, test_size: float, shuffle: str, random_state: int) -> TupleOutput:
    ...


def splitter_wrong_random_state(data: int, test_size: float, shuffle: str, random_state: str) -> TupleOutput:
    ...


@pytest.mark.parametrize(
    "splitter_fn, is_valid",
    [
        [splitter_tuple, True],
        [splitter_namedtuple, True],
        [splitter_list, True],
        [splitter_no_test_size, False],
        [splitter_no_shuffle, False],
        [splitter_no_random_state, False],
        [splitter_wrong_test_size, False],
        [splitter_wrong_shuffle, False],
        [splitter_wrong_random_state, False],
    ],
)
def test_guard_splitter(splitter_fn, is_valid):
    if is_valid:
        type_guards.guard_splitter(splitter_fn, EXPECTED_DATA_TYPE, "reader")
    else:
        with pytest.raises(TypeError):
            type_guards.guard_splitter(splitter_fn, EXPECTED_DATA_TYPE, "reader")


# parser types
Features = typing.Optional[typing.List[str]]
Targets = typing.List[str]


# valid parsers
def parser_tuple(data: int, features: Features, targets: Targets) -> TupleOutput:
    ...


def parser_namedtuple(data: int, features: Features, targets: Targets) -> NamedTupleOutput:
    ...


def parser_list(data: int, features: Features, targets: Targets) -> ListOutput:
    ...


# invalid parsers
def parser_wrong_data_type(data: str, features: Features, targets: Targets) -> TupleOutput:
    ...


def parser_no_features(data: int, targets: Targets) -> TupleOutput:
    ...


def parser_no_targets(data: int, features: Features) -> TupleOutput:
    ...


def parser_wrong_feature_type(data: int, features: str) -> TupleOutput:
    ...


def parser_wrong_target_type(data: int, targets: str) -> TupleOutput:
    ...


@pytest.mark.parametrize(
    "parser_fn, is_valid",
    [
        [parser_tuple, True],
        [parser_namedtuple, True],
        [parser_list, True],
        [parser_no_features, False],
        [parser_no_targets, False],
    ],
)
def test_guard_parser(parser_fn, is_valid):
    if is_valid:
        type_guards.guard_parser(parser_fn, EXPECTED_DATA_TYPE, "reader")
    else:
        with pytest.raises(TypeError):
            type_guards.guard_parser(parser_fn, EXPECTED_DATA_TYPE, "reader")


# trainer types
DatasetType = typing.List[float]
AnotherDatasetType = typing.List[int]


class ModelType:
    ...


class AnotherModelType:
    ...


# valid trainers
def trainer_valid(model_obj: ModelType, features: DatasetType, target: DatasetType) -> ModelType:
    ...


# invalid trainers
def trainer_wrong_model_type(model_obj: AnotherModelType, features: DatasetType, target: DatasetType) -> ModelType:
    ...


def trainer_wrong_dtype_nargs(model_obj: ModelType, data: typing.List[int]) -> ModelType:
    ...


def trainer_wrong_dtype_1(model_obj: ModelType, features: AnotherDatasetType, target: DatasetType) -> ModelType:
    ...


def trainer_wrong_dtype_2(model_obj: ModelType, features: DatasetType, target: AnotherDatasetType) -> ModelType:
    ...


def trainer_wrong_output_type(model_obj: ModelType, features: DatasetType, target: DatasetType) -> AnotherModelType:
    ...


@pytest.mark.parametrize(
    "trainer_fn, is_valid",
    [
        [trainer_valid, True],
        [trainer_wrong_model_type, False],
        [trainer_wrong_dtype_nargs, False],
        [trainer_wrong_dtype_1, False],
        [trainer_wrong_dtype_2, False],
        [trainer_wrong_output_type, False],
    ],
)
def test_guard_trainer(trainer_fn, is_valid):
    if is_valid:
        type_guards.guard_trainer(trainer_fn, ModelType, [DatasetType, DatasetType])
    else:
        with pytest.raises(TypeError):
            type_guards.guard_trainer(trainer_fn, ModelType, [DatasetType, DatasetType])


# valid evaluators
def evaluator_valid(model_obj: ModelType, features: DatasetType, target: DatasetType) -> float:
    ...


# invalid evaluators
def evaluator_wrong_model_type(model_obj: AnotherModelType, features: DatasetType, target: DatasetType) -> float:
    ...


def evaluator_wrong_dtype_nargs(model_obj: ModelType, data: DatasetType) -> float:
    ...


def evaluator_wrong_dtype_1(model_obj: ModelType, features: AnotherDatasetType, target: DatasetType) -> ModelType:
    ...


def evaluator_wrong_dtype_2(model_obj: ModelType, features: DatasetType, target: AnotherDatasetType) -> ModelType:
    ...


@pytest.mark.parametrize(
    "evaluator_fn, is_valid",
    [
        [evaluator_valid, True],
        [evaluator_wrong_model_type, False],
        [evaluator_wrong_dtype_nargs, False],
        [evaluator_wrong_dtype_1, False],
        [evaluator_wrong_dtype_2, False],
    ],
)
def test_guard_evaluator(evaluator_fn, is_valid):
    if is_valid:
        type_guards.guard_evaluator(evaluator_fn, ModelType, [DatasetType, DatasetType])
    else:
        with pytest.raises(TypeError):
            type_guards.guard_evaluator(evaluator_fn, ModelType, [DatasetType, DatasetType])


# valid predictors
def predictor_valid(model_obj: ModelType, features: DatasetType) -> float:
    ...


# invalid evaluators
def predictor_wrong_model_type(model_obj: AnotherModelType, features: DatasetType) -> float:
    ...


def predictor_wrong_dtype_nargs(model_obj: ModelType, data: DatasetType, data2: DatasetType) -> float:
    ...


def predictor_wrong_dtype(model_obj: ModelType, features: AnotherDatasetType) -> float:
    ...


def predictor_no_return_annotation(model_obj: ModelType, features: DatasetType):
    ...


@pytest.mark.parametrize(
    "predictor_fn, is_valid",
    [
        [predictor_valid, True],
        [predictor_wrong_model_type, False],
        [predictor_wrong_dtype_nargs, False],
        [predictor_wrong_dtype, False],
        [predictor_no_return_annotation, False],
    ],
)
def test_guard_predictor(predictor_fn, is_valid):
    if is_valid:
        type_guards.guard_predictor(predictor_fn, ModelType, DatasetType)
    else:
        with pytest.raises(TypeError):
            type_guards.guard_predictor(predictor_fn, ModelType, DatasetType)


# valid callbacks, assuming use of predictor_valid signature above
def callback_valid(model_obj: ModelType, features: DatasetType, prediction: float) -> None:
    ...


# invalid callbacks, same assumptions as above
def callback_wrong_model_type(model_obj: AnotherModelType, features: DatasetType, prediction: float) -> None:
    ...


def callback_wrong_prediction_dtype(model_obj: ModelType, features: DatasetType, prediction: str) -> None:
    ...


def callback_wrong_features_dtype(model_obj: ModelType, features: AnotherDatasetType, prediction: float) -> None:
    ...


def callback_wrong_missing_data(model_obj: ModelType, prediction: float) -> None:
    ...


def callback_wrong_missing_prediction(model_obj: ModelType, features: DatasetType) -> None:
    ...


def callback_wrong_dtype_nargs(
    model_obj: ModelType, features: DatasetType, prediction: float, prediction2: float
) -> None:
    ...


def callback_wrong_return_annotation(model_obj: ModelType, features: DatasetType, prediction: float) -> bool:
    ...


@pytest.mark.parametrize(
    "callback_fn, is_valid",
    [
        [callback_valid, True],
        [callback_wrong_model_type, False],
        [callback_wrong_prediction_dtype, False],
        [callback_wrong_features_dtype, False],
        [callback_wrong_missing_data, False],
        [callback_wrong_missing_prediction, False],
        [callback_wrong_dtype_nargs, False],
        [callback_wrong_return_annotation, False],
    ],
)
def test_guard_prediction_callback(callback_fn, is_valid):
    if is_valid:
        type_guards.guard_prediction_callback(callback_fn, predictor_valid, ModelType, DatasetType)
    else:
        with pytest.raises(TypeError):
            type_guards.guard_prediction_callback(callback_fn, predictor_valid, ModelType, DatasetType)


# types
UnionDatasetType = typing.Union[DatasetType, list]


# valid evaluators
def predictor_valid_union(model_obj: ModelType, features: UnionDatasetType) -> float:
    ...


def predictor_valid_inverted_order(model_obj: ModelType, features: typing.Union[list, DatasetType]) -> float:
    ...


@pytest.mark.parametrize(
    "predictor_fn, is_valid",
    [
        [predictor_valid_union, True],
        [predictor_valid_inverted_order, True],
        [predictor_wrong_dtype, False],
    ],
)
def test_guard_predictor_with_unions(predictor_fn, is_valid):
    if is_valid:
        type_guards.guard_predictor(predictor_fn, ModelType, UnionDatasetType)
    else:
        with pytest.raises(TypeError):
            type_guards.guard_predictor(predictor_fn, ModelType, UnionDatasetType)


# valid feature loaders
def feature_loader_valid(data: typing.List[float]) -> DatasetType:
    ...


def feature_loader_any(data: typing.Any) -> DatasetType:
    ...


# invalid feature loaders
def feature_loader_no_args() -> DatasetType:
    ...


def feature_loader_too_many_args(feature1: int, feature2: int) -> DatasetType:
    ...


def feature_loader_wrong_return_type(data: typing.List[int]) -> AnotherDatasetType:
    ...


@pytest.mark.parametrize(
    "fn, is_valid",
    [
        [feature_loader_valid, True],
        [feature_loader_any, True],
        [feature_loader_no_args, False],
        [feature_loader_too_many_args, False],
        [feature_loader_wrong_return_type, False],
    ],
)
def test_guard_feature_loader(fn, is_valid):
    if is_valid:
        type_guards.guard_feature_loader(fn, DatasetType)
    else:
        with pytest.raises(TypeError):
            type_guards.guard_feature_loader(fn, DatasetType)


# valid feature transformers
def feature_transformer_valid(data: DatasetType) -> DatasetType:
    ...


def feature_transformer_any(data: DatasetType) -> typing.Any:
    ...


# invalid feature transformers
def feature_transformer_no_args() -> DatasetType:
    ...


def feature_transformer_too_many_args(features1: DatasetType, features2: DatasetType) -> DatasetType:
    ...


def feature_transformer_wrong_input_type(features: int) -> DatasetType:
    ...


@pytest.mark.parametrize(
    "fn, is_valid",
    [
        [feature_transformer_valid, True],
        [feature_transformer_any, True],
        [feature_transformer_no_args, False],
        [feature_transformer_too_many_args, False],
        [feature_transformer_wrong_input_type, False],
    ],
)
def test_guard_feature_transformer(fn, is_valid):
    if is_valid:
        type_guards.guard_feature_transformer(fn, DatasetType)
    else:
        with pytest.raises(TypeError):
            type_guards.guard_feature_transformer(fn, DatasetType)
