"""Model class for defining training, evaluation, and prediction."""

from inspect import signature, Parameter
from typing import Any, Callable, Dict, NamedTuple, Type, Union

from flytekit import task, workflow


TRAIN = "train"
TEST = "test"


def train_workflow(
    model_type,
    data,
    trainer,
    evaluator,
    train_kwargs,
):
    
    def wf(model):
        train_data, test_data = data()
        trained_model = trainer(model=model, data=train_data, **train_kwargs)
        metrics = {
            TRAIN: evaluator(model=trained_model, data=train_data),
            TEST: evaluator(model=trained_model, data=test_data),
        }
        return trained_model, metrics

    wf.__signature__ = signature(wf).replace(
        parameters=(Parameter("model", annotation=model_type, kind=Parameter.KEYWORD_ONLY), ),
        return_annotation=NamedTuple("TrainingResults", model=model_type, metrics=Dict[str, float])
    )

    return workflow(wf)


class Model:
    
    def __init__(self, init: Union[Type, Callable]):
        self._init = init

    def trainer(self, fn):
        self._trainer = task(fn)
        return fn

    def predictor(self, fn):
        self._predictor = task(fn)
        return fn
    
    def evaluator(self, fn):
        self._evaluator = task(fn)
        return fn

    def train(self, hyperparameters: Dict[str, Any], data: Callable, **train_kwargs):
        train_wf = train_workflow(
            self._init,
            data,
            self._trainer,
            self._evaluator,
            train_kwargs,
        )
        return train_wf(model=self._init(**hyperparameters))

    def predict(self):
        pass
