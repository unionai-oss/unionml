"""Model class for defining training, evaluation, and prediction."""

import importlib
from enum import Enum
from fastapi import Body, HTTPException
from functools import partial, wraps
from inspect import signature, Parameter
from typing import Any, Callable, Dict, List, Optional, NamedTuple, Type, Union

from pydantic import BaseModel, create_model

from flytekit import task, workflow
from flytekit.core.workflow import WorkflowBase
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytekit.remote import FlyteRemote, FlyteWorkflowExecution
from flytekit.types.pickle import FlytePickle

from flytekit_learn.dataset import Dataset


class Endpoints(Enum):
    TRAINER = 1
    PREDICTOR = 2
    EVALUATOR = 3


def train_workflow(
    wf_name,
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

    if data is None:
        get_data = dataset()
    elif issubclass(type(data), WorkflowBase):
        get_data = data
    else:
        get_data = dataset(data=data)

    def wf(hyperparameters):
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
    wf = workflow(wf)
    wf._name = wf_name
    return wf


def predict_workflow(
    wf_name,
    dataset,
    predictor,
):
    get_features = dataset(features_only=True)

    def wf(model, features):
        return predictor(model=model, features=get_features(features=features))

    sig = signature(predictor.task_function)
    wf.__signature__ = signature(wf).replace(
        parameters=[
            Parameter("model", annotation=sig.parameters["model"].annotation, kind=Parameter.KEYWORD_ONLY),
            Parameter("features", annotation=sig.parameters["features"].annotation, kind=Parameter.KEYWORD_ONLY),
        ],
        return_annotation=sig.return_annotation,
    )
    wf = workflow(wf)
    wf._name = wf_name
    return wf


class Model:
    
    def __init__(
        self,
        name: str = None,
        *,
        init: Union[Type, Callable],
        dataset: Dataset,
        hyperparameters: Optional[Dict[str, Type]] = None
    ):
        self.name = name
        self._init_cls = init
        self._hyperparameters = hyperparameters
        self._dataset = dataset
        self._latest_model = None
        self._remote = None

        if self._dataset.name is None:
            self._dataset.name = f"{self.name}.dataset"

    def init(self, fn):
        self._init = task(fn)
        return self._init

    @property
    def train_workflow_name(self):
        return f"{self.name}.train"

    @property
    def predict_workflow_name(self):
        return f"{self.name}.predict"

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

    def train(self, hyperparameters: Dict[str, Any] = None, *, data: Any = None, lazy=False, **train_kwargs):
        train_wf = train_workflow(
            self.train_workflow_name,
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

    def predict(self, model: FlytePickle = None, features: Any = None, lazy=False):
        predict_wf = predict_workflow(
            self.predict_workflow_name,
            self._dataset,
            self._predictor,
        )
        if lazy:
            return predict_wf
        return predict_wf(model=model, features=features)

    def remote(self, config_file_path = None, project = None, domain = None):
        self._remote = FlyteRemote.from_config(
            config_file_path=config_file_path,
            default_project=project,
            default_domain=domain,
        )

    def local(self):
        self._remote = None

    def serve(self, app):

        @app.get("/")
        def main():
            return "flytekit learn: the easiest way to build and deploy models."
        
        app.get = _app_method_wrapper(app.get, self)
        app.post = _app_method_wrapper(app.post, self)
        app.put = _app_method_wrapper(app.put, self)


@Model._set_default(name="_init")
def _default_init_model(init_cls: dict, hyperparameters: dict) -> FlytePickle:
    module = importlib.import_module(init_cls["module"])
    cls = getattr(module, init_cls["cls_name"])
    return cls(**hyperparameters)


def _app_method_wrapper(app_method, model):

    @wraps(app_method)
    def wrapper(*args, **kwargs):
        return _app_decorator_wrapper(app_method(*args, **kwargs), model, app_method)

    return wrapper


def _app_decorator_wrapper(decorator, model, app_method):

    @wraps(decorator)
    def wrapper(task):

        if app_method.__name__ not in {"get", "post", "put"}:
            raise ValueError(f"flytekit-learn only supports 'get' and 'post' methods: found {app_method.__name__}")

        def _train_endpoint(
            local: bool = False,
            model_name: Optional[str] = None,
            hyperparameters: Dict = Body(..., embed=True),
        ):
            if issubclass(type(hyperparameters), BaseModel):
                hyperparameters = hyperparameters.dict()

            if not local:
                # TODO: make the model name a property of the Model object
                train_wf = model._remote.fetch_workflow(
                    name=f"{model_name}.train" if model_name else model.train_workflow_name
                )
                execution = model._remote.execute(train_wf, inputs={"hyperparameters": hyperparameters}, wait=True)
                trained_model = execution.outputs["model"]
                metrics = execution.outputs["metrics"]
            else:
                trained_model, metrics = model.train(hyperparameters=hyperparameters)
                model._latest_model = trained_model
                model._metrics = metrics
            return {
                "trained_model": str(trained_model),
                "metrics": metrics,
            }

        def _predict_endpoint(
            local: bool = False,
            model_name: Optional[str] = None,
            model_version: str = "latest",
            model_source: str = "remote",
            features: List[Dict[str, Any]] = Body(..., embed=True)
        ):
            features = model._dataset(features=features)()

            version = None if model_version == "latest" else model_version
            if model_source == "remote":
                train_wf = model._remote.fetch_workflow(
                    name=f"{model_name}.train" if model_name else model.train_workflow_name,
                    version=version,
                )
                [latest_training_execution, *_], _ = model._remote.client.list_executions_paginated(
                    train_wf.id.project,
                    train_wf.id.domain,
                    limit=1,
                    filters=[
                        filters.Equal("launch_plan.name", train_wf.id.name),
                        filters.Equal("phase", "SUCCEEDED"),
                    ],
                    sort_by=Sort("created_at", Sort.Direction.DESCENDING),
                )
                latest_training_execution = FlyteWorkflowExecution.promote_from_model(
                    latest_training_execution
                )
                model._remote.sync(latest_training_execution)
                latest_model = latest_training_execution.outputs["model"]
            else:
                if model._latest_model is None:
                    raise HTTPException(status_code=500, detail="trained model not found")
                latest_model = model._latest_model

            if not local:
                # TODO: make the model name a property of the Model object
                predict_wf = model._remote.fetch_workflow(
                    name=f"{model_name}.predict" if model_name else model.predict_workflow_name,
                    version=version,
                )
                predictions = model._remote.execute(
                    predict_wf, inputs={"model": latest_model, "features": features}, wait=True
                ).outputs["o0"]
            else:
                predictions = model.predict(model=latest_model, features=features)
            return predictions

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
                sig.parameters[p].replace(annotation=HyperparameterModel)
                if p == "hyperparameters"
                else sig.parameters[p]
                for p in sig.parameters
            ])

        decorator(endpoint_fn)
        return task

    return wrapper
