"""Unit tests for schedule module."""

import inspect
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


@pytest.mark.parametrize(
    "schedule_decorator",
    [
        schedule.schedule_trainer,
        schedule.schedule_predictor,
    ],
)
def test_schedule_decorator(schedule_decorator):
    """Test that multiple schedules can be scheduled."""
    expression = "0 * * * *"
    fixed_rate = timedelta(days=1)

    schedule_types = {
        schedule.schedule_trainer: schedule.ScheduleType.trainer,
        schedule.schedule_predictor: schedule.ScheduleType.predictor,
    }

    def fn(x: int, y: float) -> str:
        ...

    decorated_fn = schedule_decorator(name="schedule_expression", expression=expression)(fn)
    decorated_fn = schedule_decorator(name="schedule_fixed_rate", fixed_rate=fixed_rate)(decorated_fn)

    assert hasattr(fn, "__unionml_schedules__")
    assert len(fn.__unionml_schedules__) == 2
    assert inspect.signature(fn) == inspect.signature(decorated_fn)
    for name, s in zip(("schedule_expression", "schedule_fixed_rate"), fn.__unionml_schedules__):
        assert isinstance(s, schedule.Schedule)
        assert name == s.name
        assert s.type is schedule_types[schedule_decorator]


class MockModel:
    def __init__(self):
        self.training_schedules = []
        self.prediction_schedules = []

    def add_trainer_schedule(self, schedule: schedule.Schedule):
        self.training_schedules.append(schedule)

    def add_predictor_schedule(self, schedule: schedule.Schedule):
        self.prediction_schedules.append(schedule)


@pytest.mark.parametrize(
    "schedule_decorator",
    [
        schedule.schedule_trainer,
        schedule.schedule_predictor,
    ],
)
def test_schedule_decorator_from_model(schedule_decorator):
    """
    Test that multiple trainer schedules can be specified with a function containing the special __unionml_model__
    attribute.
    """
    expression = "0 * * * *"
    fixed_rate = timedelta(days=1)

    schedule_attrs = {
        schedule.schedule_trainer: "training_schedules",
        schedule.schedule_predictor: "prediction_schedules",
    }

    schedule_types = {
        schedule.schedule_trainer: schedule.ScheduleType.trainer,
        schedule.schedule_predictor: schedule.ScheduleType.predictor,
    }

    def fn(x: int, y: float) -> str:
        ...

    # emulate using the scheduling decorator after a function is decorated with @model.trainer or @model.predictor
    model = MockModel()
    fn.__unionml_model__ = model
    decorated_fn = schedule_decorator(name="schedule_expression", expression=expression)(fn)
    decorated_fn = schedule_decorator(name="schedule_fixed_rate", fixed_rate=fixed_rate)(decorated_fn)

    schedules = getattr(model, schedule_attrs[schedule_decorator])
    assert len(schedules) == 2
    assert inspect.signature(fn) == inspect.signature(decorated_fn)
    for name, s in zip(("schedule_expression", "schedule_fixed_rate"), schedules):
        assert isinstance(s, schedule.Schedule)
        assert name == s.name
        assert s.type is schedule_types[schedule_decorator]
