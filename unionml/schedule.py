"""Module to handle scheduling launchplans."""

from datetime import timedelta
from typing import Optional

from flytekit import CronSchedule, FixedRate, LaunchPlan
from flytekit.core.workflow import WorkflowBase


def create_scheduled_launchplan(
    workflow: WorkflowBase,
    name: str,
    *,
    expression: Optional[str] = None,
    offset: Optional[str] = None,
    fixed_rate: Optional[timedelta] = None,
    time_arg: Optional[str] = None,
    **kwargs,
) -> LaunchPlan:
    """Create a :class:`~flytekit.LaunchPlan` with a schedule.

    :param workflow: UnionML-derived workflow.
    :param name: unique name of the launch plan
    :param expression: a cron expression (see
        `here <https://docs.flyte.org/en/latest/concepts/schedules.html#cron-expression>`__)
        or valid croniter schedule for e.g. `@daily`, `@hourly`, `@weekly`, `@yearly`
        (see `here <https://github.com/kiorky/croniter#keyword-expressions>`__).
    :param offset: duration to offset the schedule, must be a valid ISO 8601
        duration. Only used if ``expression`` is specified.
    :param fixed_rate: a :class:`~datetime.timedelta` object representing fixed
        rate with which to run the workflow.
    :param time_arg: the name of the argument in the ``workflow`` that will receive
        the kickoff time of the scheduled launchplan.
    :param kwargs: additional keyword arguments to pass to
        :class:`~flytekit.LaunchPlan`
    :returns: a scheduled launch plan object.
    """
    if expression is not None and fixed_rate is not None:
        raise ValueError("You must specify exactly one of 'expression' or 'fixed_rate', not both.")
    elif expression:
        schedule = CronSchedule(
            schedule=expression,
            offset=offset,
            kickoff_time_input_arg=time_arg,
        )
    elif fixed_rate:
        schedule = FixedRate(
            duration=fixed_rate,
            kickoff_time_input_arg=time_arg,
        )
    else:
        raise ValueError("You must specify exactly one of 'expression' or 'fixed_rate'.")
    return LaunchPlan.get_or_create(
        name=name,
        workflow=workflow,
        schedule=schedule,
        **kwargs,
    )
