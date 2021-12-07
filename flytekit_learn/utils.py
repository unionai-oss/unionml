from functools import partial, wraps
from inspect import signature, Parameter

from flytekit import task

from flytekit_learn.task_resolver import task_resolver


def inner_task(
    fn=None,
    *,
    fklearn_obj,
    input_parameters=None,
    return_annotation=None,
    **task_kwargs,
):
    """A flytekit task defined within a Dataset or Model class.
    
    This wrapper does the following:
    - makes sure the wrapper function has the same signature as the origin function.
    - renames the wrapped function to ``task_name``.
    - assigns an ``fklearn_obj`` to the function object.
    - converts the wrapper function into a flytekit task, using the flytekit-learn task resolver.
    """

    if fn is None:
        return partial(
            inner_task,
            fklearn_obj=fklearn_obj,
            input_parameters=input_parameters,
            return_annotation=return_annotation,
            **task_kwargs,
        ) 

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    fn_sig = signature(fn)
    wrapper.__signature__ = signature(wrapper).replace(
        parameters=[
            p.replace(kind=Parameter.KEYWORD_ONLY) for p in
            (fn_sig.parameters if input_parameters is None else input_parameters).values()
        ],
        return_annotation=fn_sig.return_annotation if return_annotation is None else return_annotation,
    )
    wrapper.__fklearn_object__ = fklearn_obj
    output_task = task(wrapper, task_resolver=task_resolver)
    output_task._name = f"{fklearn_obj.name}.{wrapper.__name__}"
    return output_task
