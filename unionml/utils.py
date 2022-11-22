import importlib
import typing
from functools import partial, wraps
from inspect import Parameter, _empty, signature

from flytekit import task

from unionml.task_resolver import task_resolver


def inner_task(
    fn=None,
    *,
    unionml_obj,
    input_parameters=None,
    return_annotation=_empty,
    **task_kwargs,
):
    """A flytekit task defined within a Dataset or Model class.

    This wrapper does the following:
    - makes sure the wrapper function:
      - has the same signature as the original function
      - OR it takes on the signature specified by the ``input_parameters`` and ``return_annotation`` arguments
        if they are provided
    - renames the wrapped function to ``task_name``.
    - assigns an ``unionml_obj`` to the function object.
    - converts the wrapper function into a flytekit task, using the unionml task resolver.
    """

    if fn is None:
        return partial(
            inner_task,
            unionml_obj=unionml_obj,
            input_parameters=input_parameters,
            return_annotation=return_annotation,
            **task_kwargs,
        )

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    fn_sig = signature(fn)

    # update signature and type annotations of wrapped function
    wrapper.__signature__ = signature(wrapper).replace(
        parameters=[
            p.replace(kind=Parameter.KEYWORD_ONLY)
            for p in (fn_sig.parameters if input_parameters is None else input_parameters).values()
        ],
        return_annotation=fn_sig.return_annotation if return_annotation is _empty else return_annotation,
    )
    wrapper.__annotations__.update({k: v.annotation for k, v in input_parameters.items()})
    wrapper.__annotations__["return"] = return_annotation

    wrapper.__unionml_object__ = unionml_obj
    output_task = task(wrapper, task_resolver=task_resolver, **task_kwargs)
    output_task._name = f"{unionml_obj.name}.{wrapper.__name__}"
    return output_task


def is_pytorch_model(model_type: typing.Type):
    return model_type.__bases__[0].__module__.startswith("torch") or model_type.__module__.startswith("torch")


def is_keras_model(model_type: typing.Type):
    return model_type.__bases__[0].__module__.startswith("keras")


def module_is_installed(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False
