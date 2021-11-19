"""Model class for defining training, evaluation, and prediction."""

from functools import partial, wraps
import importlib
from inspect import signature, Parameter
from typing import Any, Callable, Dict, NamedTuple, Tuple, Type, Union

from flytekit import task, workflow, Workflow
from flytekit.types.pickle import FlytePickle


def train_workflow_imperative(
    model_type,
    data,
    trainer,
    evaluator,
    train_kwargs,
):
    wf = Workflow("train_workflow")
    wf.add_workflow_input("model", FlytePickle)
    data_node = wf.add_entity(data)
    trainer_node = wf.add_entity(trainer, model=wf.inputs["model"], data=data_node.outputs["train"], **train_kwargs)
    train_eval_node = wf.add_entity(evaluator, model=trainer_node.outputs["o0"], data=data_node.outputs["train"])
    test_eval_node = wf.add_entity(evaluator, model=trainer_node.outputs["o0"], data=data_node.outputs["test"])
    wf.add_workflow_output("model", trainer_node.outputs["o0"])
    wf.add_workflow_output(
        "metrics",
        {
            "train": train_eval_node.outputs["o0"],
            "test": test_eval_node.outputs["o0"],
        },
        python_type=Dict[str, evaluator.python_interface.outputs["o0"]],
    )
    return wf


def train_workflow(
    init,
    data,
    trainer,
    evaluator,
    train_kwargs,
    init_cls=None,
):
    init_kwargs = {}
    if init_cls:
        init_kwargs["init_cls"] = {
            "module": init_cls.__module__,
            "cls_name": init_cls.__name__,
        }

    def wf(hyperparameters: dict):
        train_data, test_data = data()
        trained_model = trainer(
            model=init(hyperparameters=hyperparameters, **init_kwargs), data=train_data, **train_kwargs
        )
        metrics = {
            "train": evaluator(model=trained_model, data=train_data),
            "test": evaluator(model=trained_model, data=test_data),
        }
        return trained_model, metrics

    wf.__signature__ = signature(wf).replace(
        parameters=(Parameter("hyperparameters", annotation=dict, kind=Parameter.KEYWORD_ONLY), ),
        return_annotation=NamedTuple(
            "TrainingResults", model=trainer.python_interface.outputs["o0"], metrics=Dict[str, float]
        )
    )

    return workflow(wf)


class Model:
    
    def __init__(self, init: Union[Type, Callable]):
        self._init_cls = init

    def init(self, fn):
        self._init = task(fn)
        return self._init

    @classmethod
    def _set_default(cls, fn=None, *, name):
        if fn is None:
            return partial(cls._set_default, name=name)

        setattr(cls, name, task(fn))
        return getattr(cls, name)

    def trainer(self, fn):
        self._trainer = task(fn)
        return self._trainer

    def predictor(self, fn):
        self._predictor = task(fn)
        return self._predictor
    
    def evaluator(self, fn):
        self._evaluator = task(fn)
        return self._evaluator

    def train(self, hyperparameters: Dict[str, Any], data: Callable, **train_kwargs):
        train_wf = train_workflow(
            self._init,
            data,
            self._trainer,
            self._evaluator,
            train_kwargs,
            init_cls=self._init_cls,
        )
        return train_wf(hyperparameters=hyperparameters)

    def train_workflow(self, data: Callable, **train_kwargs):
        return train_workflow(
            self._init,
            data,
            self._trainer,
            self._evaluator,
            train_kwargs,
            init_cls=self._init_cls
        )

    def predict(self):
        pass


# task stub for initializing the model
@Model._set_default(name="_init")
def _default_init_model(init_cls: dict, hyperparameters: dict) -> FlytePickle:
    module = importlib.import_module(init_cls["module"])
    cls = getattr(module, init_cls["cls_name"])
    return cls(**hyperparameters)
