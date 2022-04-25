from functools import partial, wraps
from inspect import Parameter, signature

from flytekit import task

from ulearn.task_resolver import task_resolver


def inner_task(
    fn=None,
    *,
    ulearn_obj,
    input_parameters=None,
    return_annotation=None,
    **task_kwargs,
):
    """A flytekit task defined within a Dataset or Model class.

    This wrapper does the following:
    - makes sure the wrapper function:
      - has the same signature as the original function
      - OR it takes on the signature specified by the ``input_parameters`` and ``return_annotation`` arguments
        if they are provided
    - renames the wrapped function to ``task_name``.
    - assigns an ``ulearn_obj`` to the function object.
    - converts the wrapper function into a flytekit task, using the ulearn task resolver.
    """

    if fn is None:
        return partial(
            inner_task,
            ulearn_obj=ulearn_obj,
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
        return_annotation=fn_sig.return_annotation if return_annotation is None else return_annotation,
    )
    wrapper.__annotations__.update({k: v.annotation for k, v in input_parameters.items()})
    wrapper.__annotations__["return"] = return_annotation

    wrapper.__ulearn_object__ = ulearn_obj
    output_task = task(wrapper, task_resolver=task_resolver, **task_kwargs)
    output_task._name = f"{ulearn_obj.name}.{wrapper.__name__}"
    return output_task
