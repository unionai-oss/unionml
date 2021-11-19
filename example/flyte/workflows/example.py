"""
Hello World
------------
This simple workflow calls a task that returns "Hello World" and then just sets that as the final output of the workflow.
"""
import typing

# %%
# All imports at the root flytekit level are stable and we maintain backwards
# compatibility for them.
from flytekit import task, workflow


# %%
# Here we define a task called ``say_hello``. Note the @task decorator, Flyte
# uses this to understand that you intend to port this function to flyte.
# If you have normal functions in this file, they are not accessible to
# file, unless they have the @task decorator.
# You can change the signature of the task to take in an argument like this:
# def say_hello(name: str) -> str:
@task
def say_hello() -> str:
    return "hello world"


# %%
# Here we declare a workflow called ``my_wf``. Note the @workflow decorator,
# Flyte finds all workflows that you have declared by finding this decorator.
# A @workflow function, looks like a regular python function, except for some
# important differences, it is never executed by flyte-engine. It is like
# psuedo code, that is analyzed by flytekit to convert to Flyte's native
# Workflow representation. Thus the variables like return values from `tasks`
# are not real values, and trying to interact with them like regular variables
# will result in an error. For example, if a task returns a boolean, and if you
# try to test the truth value for this boolean, an error will be raised. The
# reason, is the tasks are not really executed by the function, but run remote
# and the return variables are supplied to subsequent tasks.
#
# You can treat the outputs of a task as you normally would a Python function. Assign the output to two variables
# and use them in subsequent tasks as normal. See :py:func:`flytekit.workflow`
# You can change the signature of the workflow to take in an argument like this:
# def my_wf(name: str) -> str:
@workflow
def my_wf() -> str:
    res = say_hello()
    return res


# %%
# Execute the Workflow, simply by invoking it like a function and passing in
# the necessary parameters
#
# .. note::
#
#   One thing to remember, currently we only support ``Keyword arguments``. So
#   every argument should be passed in the form ``arg=value``. Failure to do so
#   will result in an error
if __name__ == "__main__":
    print(f"Running my_wf() { my_wf() }")
