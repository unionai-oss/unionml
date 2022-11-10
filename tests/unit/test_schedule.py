"""Unit tests for schedule module."""

from datetime import datetime, timedelta

import pytest
from flytekit import workflow

from unionml import schedule


@workflow
def wf_no_args():
    ...


@workflow
def wf_with_args(arg1: int, arg2: str):
    ...


@workflow
def wf_with_time_arg(time_arg: datetime):
    ...


@pytest.mark.parametrize(
    "wf, time_arg",
    [
        [wf_no_args, None],
        [wf_with_args, None],
        [wf_with_time_arg, "time_arg"],
    ],
)
def test_create_scheduled_launchplan_expression(wf, time_arg):
    """Test creating a scheduled launchplan with an expression."""
    name = f"{wf.short_name}_schedule_expression"
    expression = "0 * * * *"
    lp_expression = schedule.create_scheduled_launchplan(
        wf,
        name,
        expression=expression,
        time_arg=time_arg,
    )
    assert lp_expression.name == name
    assert lp_expression.schedule.schedule_expression.schedule == expression


@pytest.mark.parametrize(
    "wf, time_arg",
    [
        [wf_no_args, None],
        [wf_with_args, None],
        [wf_with_time_arg, "time_arg"],
    ],
)
def test_create_scheduled_launchplan_fixed_rate(wf, time_arg):
    """Test creating a scheduled launchplan with a fixed rate."""
    name = f"{wf.short_name}_schedule_fixed_rate"
    fixed_rate = timedelta(days=1)
    lp_fixed_rate = schedule.create_scheduled_launchplan(
        wf,
        name,
        fixed_rate=fixed_rate,
        time_arg=time_arg,
    )
    assert lp_fixed_rate.name == name
    assert lp_fixed_rate.schedule.rate.unit == 2  # day
    assert lp_fixed_rate.schedule.rate.value == 1


def test_create_scheduled_launchplan_exceptions():
    """Test exception-raising execution paths for create_scheduled_launchplan."""
    name = f"{wf_no_args.short_name}_schedule_exception"
    with pytest.raises(
        ValueError,
        match="You must specify exactly one of 'expression' or 'fixed_rate', not both.",
    ):
        schedule.create_scheduled_launchplan(
            wf_no_args,
            name,
            expression="*/1 * * * *",
            fixed_rate=timedelta(days=1),
        )

    with pytest.raises(ValueError, match="Schedule is invalid."):
        schedule.create_scheduled_launchplan(
            wf_no_args,
            name,
            expression="foobar",
        )

    with pytest.raises(AttributeError, match="'str' object has no attribute 'microseconds'"):
        schedule.create_scheduled_launchplan(
            wf_no_args,
            name,
            fixed_rate="foobar",
        )

    with pytest.raises(
        ValueError,
        match="You must specify exactly one of 'expression' or 'fixed_rate'",
    ):
        schedule.create_scheduled_launchplan(wf_no_args, name)
