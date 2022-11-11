"""Module to handle scheduling training and batch prediction jobs."""

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Optional, Union

from flytekit import CronSchedule, FixedRate, LaunchPlan
from flytekit.core.workflow import WorkflowBase


class ScheduleType(Enum):
    """Allowable schedule types."""

    trainer = "trainer"
    """Indicates a schedule that kicks of a training run."""

    predictor = "predictor"
    """Indicates a schedule that kicks of a prediction run."""


@dataclass
class Schedule:
    """Data model for cron expression and fixed-rate schedules."""

    type: Union[str, ScheduleType]
    """'trainer' or 'predictor' schedule type."""

    name: str
    """Name of the schedule. Must be unique in the context of a Model definiton."""

    expression: Optional[str] = None
    """
    A `cron expression <https://docs.flyte.org/en/latest/concepts/schedules.html#cron-expression>`__)
    or valid `croniter schedule <https://github.com/kiorky/croniter#keyword-expressions>`__ for e.g.
    ``"@daily"``, ``"@hourly"``, ``"@weekly"``, ``"@yearly"``.
    """

    offset: Optional[str] = None
    """
    Duration to offset the schedule, must be a valid
    `ISO 8601 duration <https://en.wikipedia.org/wiki/ISO_8601>`__ . Only
    used if ``expression`` is specified.
    """

    fixed_rate: Optional[timedelta] = None
    """A :class:`~datetime.timedelta` object representing fixed rate with which to run the workflow."""

    time_arg: Optional[str] = None
    """The name of the argument in the ``workflow`` that will receive the kickoff time of the scheduled launchplan."""

    inputs: Optional[dict] = None
    """Inputs to be passed into the scheduled launchplan."""

    activate_on_deploy: bool = True
    """Whether or not to automatically activate this schedule on app deployment."""

    launchplan_kwargs: Optional[dict] = None
    """Additional keyword arguments to pass to :class:`flytekit.LaunchPlan`"""

    def __post_init__(self):
        """Handles post-initialization of the instance."""
        if isinstance(self.type, str):
            self.type = ScheduleType[self.type]


def create_scheduled_launchplan(
    workflow: WorkflowBase,
    name: str,
    *,
    expression: Optional[str] = None,
    offset: Optional[str] = None,
    fixed_rate: Optional[timedelta] = None,
    time_arg: Optional[str] = None,
    inputs: Optional[dict] = None,
    **launchplan_kwargs,
) -> LaunchPlan:
    """Create a :class:`~flytekit.LaunchPlan` with a schedule.

    :param workflow: UnionML-derived workflow.
    :param name: unique name of the launch plan
    :param expression: A `cron expression <https://docs.flyte.org/en/latest/concepts/schedules.html#cron-expression>`__)
        or valid `croniter schedule <https://github.com/kiorky/croniter#keyword-expressions>`__ for e.g.
        ``"@daily"``, ``"@hourly"``, ``"@weekly"``, ``"@yearly"``.
    :param offset: duration to offset the schedule, must be a
        valid `ISO 8601 duration <https://en.wikipedia.org/wiki/ISO_8601>`__ . Only used if ``expression`` is specified.
    :param fixed_rate: a :class:`~datetime.timedelta` object representing fixed
        rate with which to run the workflow.
    :param time_arg: the name of the argument in the ``workflow`` that will receive
        the kickoff time of the scheduled launchplan.
    :param kwargs: additional keyword arguments to pass to
        :class:`flytekit.LaunchPlan`
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

    inputs = inputs or {}
    if "fixed_inputs" in launchplan_kwargs:
        update_inputs = launchplan_kwargs.pop("fixed_inputs")
        inputs.update(update_inputs)

    return LaunchPlan.get_or_create(
        name=name,
        workflow=workflow,
        schedule=schedule,
        fixed_inputs=inputs,
        **launchplan_kwargs,
    )
