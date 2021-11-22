"""Model class for defining training, evaluation, and prediction."""

import importlib
from enum import Enum
from fastapi import HTTPException
from functools import partial, wraps
from inspect import signature, Parameter
from typing import Any, Callable, Dict, List, Optional, NamedTuple, Tuple, Type, Union

from pydantic import BaseModel, create_model

from flytekit import task, workflow
from flytekit.core.workflow import WorkflowBase
from flytekit.types.pickle import FlytePickle

from flytekit_learn.dataset import Dataset


class Endpoints(Enum):
    TRAINER = 1
    PREDICTOR = 2
    EVALUATOR = 3


def train_workflow(
    data,
    dataset,
    init,
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

    if not issubclass(type(data), WorkflowBase):
        get_data = dataset(data=data)
    else:
        get_data = data

    def wf(hyperparameters: dict):
        train_data, test_data = get_data()
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


def predict_workflow(
    trained_model,
    dataset,
    features,
    predictor,
):
    if not issubclass(type(features), WorkflowBase):
        get_features = dataset(features=features)
    else:
        get_features = features

    def wf():
        data = get_features()
        return predictor(model=trained_model, features=data)

    wf.__signature__ = signature(wf).replace(
        return_annotation=signature(predictor.task_function).return_annotation
    )

    return workflow(wf)


class Model:
    
    def __init__(
        self,
        init: Union[Type, Callable],
        dataset: Dataset,
        hyperparameters: Optional[Dict[str, Type]] = None
    ):
        self._init_cls = init
        self._hyperparameters = hyperparameters
        self._dataset = dataset
        self._latest_model = None

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
        self._trainer.__app_method__ = Endpoints.TRAINER
        return self._trainer

    def predictor(self, fn):
        self._predictor = task(fn)
        self._predictor.__app_method__ = Endpoints.PREDICTOR
        return self._predictor
    
    def evaluator(self, fn):
        self._evaluator = task(fn)
        return self._evaluator

    def train(self, hyperparameters: Dict[str, Any] = None, *, data: Any, lazy=False, **train_kwargs):
        train_wf = train_workflow(
            data,
            self._dataset,
            self._init,
            self._trainer,
            self._evaluator,
            train_kwargs,
            init_cls=self._init_cls,
        )
        if lazy:
            return train_wf
        if hyperparameters is None:
            raise ValueError("hyperparameters need to be provided when eager=True")

        trained_model, metrics = train_wf(hyperparameters=hyperparameters)
        self._latest_model = trained_model
        self._latest_metrics = metrics
        return trained_model, metrics

    def predict(self, trained_model: FlytePickle, features: Any, lazy=False):
        predict_wf = predict_workflow(
            trained_model,
            self._dataset,
            features,
            self._predictor,
        )
        if lazy:
            return predict_wf
        return predict_wf()

    def serve(self, app):
        app.get = _app_method_wrapper(app.get, self, self._dataset)
        app.post = _app_method_wrapper(app.post, self, self._dataset)
        app.put = _app_method_wrapper(app.put, self, self._dataset)


@Model._set_default(name="_init")
def _default_init_model(init_cls: dict, hyperparameters: dict) -> FlytePickle:
    module = importlib.import_module(init_cls["module"])
    cls = getattr(module, init_cls["cls_name"])
    return cls(**hyperparameters)


def _app_method_wrapper(app_method, model, dataset):

    @wraps(app_method)
    def wrapper(*args, **kwargs):
        return _app_decorator_wrapper(app_method(*args, **kwargs), model, dataset, app_method)

    return wrapper


def _app_decorator_wrapper(decorator, model, dataset, app_method):

    @wraps(decorator)
    def wrapper(task):

        if app_method.__name__ not in {"get", "post", "put"}:
            raise ValueError(f"flytekit-learn only supports 'get' and 'post' methods: found {app_method.__name__}")

        def _train_endpoint(hyperparameters: Dict, data: List[Dict[str, Any]]):
            if issubclass(type(hyperparameters), BaseModel):
                hyperparameters = hyperparameters.dict()
            trained_model, metrics = model.train(hyperparameters=hyperparameters, data=data)
            model._latest_model = trained_model
            model._metrics = metrics
            return {"training_score": metrics}

        def _predict_endpoint(kwargs: dict):
            if model._latest_model is None:
                raise HTTPException(status_code=500, detail="trained model not found")
            features_type = model._predictor.python_interface.inputs["features"]
            data = features_type(kwargs["features"])
            features, _ = dataset._parser.task_function(data, dataset._features, dataset._targets)
            return task(model=model._latest_model, features=features)

        endpoint_fn = {
            Endpoints.PREDICTOR: _predict_endpoint,
            Endpoints.TRAINER: _train_endpoint,
        }[task.__app_method__]

        if endpoint_fn is _train_endpoint and model._hyperparameters:
            HyperparameterModel = create_model(
                "HyperparameterModel", **{k: (v, ...) for k, v in model._hyperparameters.items()}
            )
            sig = signature(_train_endpoint)
            _train_endpoint.__signature__ = sig.replace(parameters=[
                sig.parameters["hyperparameters"].replace(annotation=HyperparameterModel),
                sig.parameters["data"],
            ])

        return decorator(endpoint_fn)

    return wrapper
