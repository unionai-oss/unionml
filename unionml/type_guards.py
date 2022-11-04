"""Type checking utilities for core function decorators."""

from inspect import Parameter, _empty, signature
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Type

try:
    from typing import get_args, get_origin  # type: ignore
except ImportError:
    from typing_extensions import get_args, get_origin


SPLITTER_KWTYPES: Dict[str, object] = {
    "test_size": float,
    "shuffle": bool,
    "random_state": int,
}

PARSER_KWTYPES: Dict[str, object] = {
    "features": Optional[List[str]],
    "targets": List[str],
}


def _is_tuple_or_list_type(type: Type):
    return get_origin(type) in {tuple, list} or getattr(type, "__bases__", None) == (tuple,)


def _check_input_data_type(fn_name: str, actual_type: Type, expected_type: Type):
    if actual_type is Any or expected_type is Any:
        return

    if (
        actual_type != expected_type
        and expected_type not in get_args(actual_type)
        and actual_type not in get_args(expected_type)
    ):
        raise TypeError(
            f"The type of the first argument of the '{fn_name}' function must be compatible with the expected output "
            f"type: {expected_type}. Found {actual_type}"
        )


def _check_supported_generic_type(fn_name: str, type: Type):
    if not _is_tuple_or_list_type(type):
        raise TypeError(
            f"The output of '{fn_name}' must be a List, Tuple, or NamedTuple type containing data splits. "
            f"Found {type}"
        )


def _check_generic_arg_types(fn_name: str, generic_type: Type, expected_type: Type, expected_type_source: str):
    for subtype in get_args(generic_type):
        if subtype != expected_type:
            raise TypeError(
                f"The type arguments to the output generic type of '{fn_name}' the function must match the "
                f"'{expected_type_source}' output type: {expected_type}. Found {generic_type}"
            )


def _check_parameters(fn_name: str, parameters: Mapping[str, Parameter], kwtypes: Dict[str, object]):
    for i, (argname, argtype) in enumerate(kwtypes.items()):
        param = parameters.get(argname)
        if param is None:
            raise TypeError(
                f"The '{fn_name}' function is expected to accept an argument '{argname}' of type {argtype} "
                f"at the {i + 1}th position. Found a function with the following signature: {parameters}"
            )
        if param.annotation != argtype:
            raise TypeError(f"The argument '{argname}' expected to be of type {argtype}, found {param.annotation}")


def _check_data_types_length(actual_types, expected_types):
    if len(actual_types) != len(expected_types):
        raise TypeError(
            f"Length of positional data arguments are expected to match {expected_types}. Found {actual_types}."
        )


def guard_reader(reader):
    """Ensure that reader return annotation is not empty."""
    reader_sig = signature(reader)
    if reader_sig.return_annotation is _empty:
        raise TypeError(
            "The dataset.reader function return annotation cannot be empty. You need to specify a return type."
        )


def guard_loader(loader: Callable, expected_data_type: Type):
    """Ensure that the first arg of the loader is of the expected type."""
    sig = signature(loader)
    actual_data_type = [*sig.parameters.values()][0].annotation
    _check_input_data_type("loader", actual_data_type, expected_data_type)


def guard_splitter(splitter: Callable, expected_data_type: Type, expected_type_source: str):
    """Ensure that the splitter has the expected input data type."""
    sig = signature(splitter)
    actual_data_type = [*sig.parameters.values()][0].annotation
    output_type = sig.return_annotation

    _check_input_data_type("splitter", actual_data_type, expected_data_type)
    _check_supported_generic_type("splitter", output_type)
    _check_generic_arg_types("splitter", output_type, expected_data_type, expected_type_source)
    _check_parameters("splitter", sig.parameters, SPLITTER_KWTYPES)


def guard_parser(parser: Callable, expected_data_type: Type, expected_type_source: str):
    """Ensure that the parser has the expected input data type."""
    sig = signature(parser)
    actual_data_type = [*sig.parameters.values()][0].annotation
    output_type = sig.return_annotation

    _check_input_data_type("parser", actual_data_type, expected_data_type)
    _check_supported_generic_type("parser", output_type)
    _check_parameters("parser", sig.parameters, PARSER_KWTYPES)


def guard_trainer(trainer: Callable, expected_model_type: Type, expected_data_types: Iterable[Type]):
    """Ensure that the trainer has the expected input data and model types."""
    sig = signature(trainer)
    params = [*sig.parameters.values()]

    actual_model_type = params[0].annotation
    actual_data_types = [
        p.annotation for p in params[1:] if p.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY}
    ]

    _check_input_data_type("trainer", actual_model_type, expected_model_type)
    _check_input_data_type("trainer", sig.return_annotation, expected_model_type)
    _check_data_types_length(actual_data_types, expected_data_types)
    for actual_dtype, expected_dtype in zip(actual_data_types, expected_data_types):
        _check_input_data_type("trainer", actual_dtype, expected_dtype)


def guard_evaluator(evaluator: Callable, expected_model_type: Type, expected_data_types: Iterable[Type]):
    """Ensure that the evaluater has the expected input data and model types."""
    sig = signature(evaluator)
    params = [*sig.parameters.values()]

    actual_model_type = params[0].annotation
    actual_data_types = [
        p.annotation for p in params[1:] if p.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY}
    ]

    _check_input_data_type("evaluator", actual_model_type, expected_model_type)
    _check_data_types_length(actual_data_types, expected_data_types)
    for actual_dtype, expected_dtype in zip(actual_data_types, expected_data_types):
        _check_input_data_type("evaluator", actual_dtype, expected_dtype)


def guard_predictor(predictor: Callable, expected_model_type: Type, expected_data_type: Type):
    """Ensure that the predictor has the expected input data and model types."""
    sig = signature(predictor)
    params = [*sig.parameters.values()]

    actual_model_type = params[0].annotation
    actual_data_types = [
        p.annotation for p in params[1:] if p.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY}
    ]

    if len(actual_data_types) != 1:
        raise TypeError(f"The 'predictor' function must take a single 'features' argument, found {actual_data_types}")

    actual_data_type = actual_data_types[0]
    _check_input_data_type("predictor", actual_model_type, expected_model_type)
    _check_input_data_type("predictor", actual_data_type, expected_data_type)

    if sig.return_annotation is _empty:
        raise TypeError("The 'predictor' function needs a return type annotation.")


def guard_prediction_callback(
    callback: Callable, predictor: Callable, expected_model_type: Type, expected_data_type: Type
):
    """Ensure that a callback has the expected model type, along with input and output data types."""
    sig = signature(callback)
    params = [*sig.parameters.values()]

    expected_prediction_type = signature(predictor).return_annotation
    if expected_prediction_type is _empty:
        raise TypeError("The 'predictor' function needs a return type annotation.")

    if sig.return_annotation is not _empty and sig.return_annotation is not None:
        raise TypeError(f"The 'callback[{callback.__name__}]' function must have None as it's return annotation.")

    actual_model_type = params[0].annotation
    actual_params = [
        p.annotation for p in params[1:] if p.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY}
    ]

    if len(actual_params) != 2:
        raise TypeError(
            f"Callback functions (callback[{callback.__name__}]) must take both 'features' and 'prediction' arguments, found {actual_params}"
        )

    actual_features, actual_prediction = actual_params

    if (
        actual_model_type is not Any
        and expected_model_type is not Any
        and actual_model_type != expected_model_type
        and expected_model_type not in get_args(actual_model_type)
        and actual_model_type not in get_args(expected_model_type)
    ):
        raise TypeError(
            f"The type of the first argument of the callback[{callback.__name__}] function must be compatible with the expected output "
            f"type: {expected_model_type}. Found {actual_model_type}"
        )

    if (
        actual_features is not Any
        and expected_data_type is not Any
        and actual_features != expected_data_type
        and expected_data_type not in get_args(actual_features)
        and actual_features not in get_args(expected_data_type)
    ):
        raise TypeError(
            f"The type of the second argument of the callback[{callback.__name__}] function must be compatible with the expected output "
            f"type: {expected_data_type}. Found {actual_features}"
        )

    if (
        actual_prediction is not Any
        and expected_prediction_type is not Any
        and actual_prediction != expected_prediction_type
        and expected_prediction_type not in get_args(actual_prediction)
        and actual_prediction not in get_args(expected_prediction_type)
    ):
        raise TypeError(
            f"The type of the third argument of the callback[{callback.__name__}] function must be compatible with the expected output "
            f"type: {expected_prediction_type}. Found {actual_prediction}"
        )


def guard_feature_loader(feature_loader: Callable, expected_data_type: Type):
    """Ensure that the feature loader return type needs to match the parser data input."""
    sig = signature(feature_loader)
    params = [*sig.parameters.values()]
    if len(sig.parameters) != 1:
        raise TypeError(
            "The 'feature_loader' must take a single argument representing raw features or a reference to raw features."
        )
    actual_data_type = params[0].annotation
    _check_input_data_type("feature_loader", actual_data_type, expected_data_type)


def guard_feature_transformer(feature_transformer: Callable, expected_data_type: Type):
    """Ensure that the feature_transformer input matches the return type of parser."""
    sig = signature(feature_transformer)
    params = [*sig.parameters.values()]
    if len(sig.parameters) != 1:
        raise TypeError("The 'feature_transformer' must take a single argument representing the loaded features.")
    actual_data_type = params[0].annotation
    _check_input_data_type("feature_transformer", actual_data_type, expected_data_type)
