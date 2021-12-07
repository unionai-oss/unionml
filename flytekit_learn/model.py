"""Model class for defining training, evaluation, and prediction."""

import importlib
from collections import OrderedDict
from enum import Enum
from fastapi import Body, HTTPException
from functools import partial, wraps
from inspect import signature, Parameter
from typing import Any, Callable, Dict, List, Optional, NamedTuple, Type, Union

from pydantic import BaseModel, create_model

from flytekit import task, workflow, Workflow
from flytekit.core.tracker import TrackedInstance
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytekit.remote import FlyteRemote, FlyteWorkflowExecution
from flytekit.types.pickle import FlytePickle

from flytekit_learn.dataset import Dataset
from flytekit_learn.utils import inner_task


class Endpoints(Enum):
    TRAINER = 1
    PREDICTOR = 2
    EVALUATOR = 3


class Model(TrackedInstance):
    
    def __init__(
        self,
        name: str = None,
        *,
        init: Union[Type, Callable],
        dataset: Dataset,
        hyperparameters: Optional[Dict[str, Type]] = None,
    ):
        super().__init__()
        self.name = name
        self._init_cls = init
        self._hyperparameters = hyperparameters
        self._dataset = dataset
        self._latest_model = None
        self._remote = None

        if self._dataset.name is None:
            self._dataset.name = f"{self.name}.dataset"

        self._train_task = None
        self._predict_task = None
        self._predict_from_features_task = None
        self._train_task_kwargs = None
        self._predict_task_kwargs = None

    @property
    def train_workflow_name(self):
        return f"{self.name}.train"

    @property
    def predict_workflow_name(self):
        return f"{self.name}.predict"

    @property
    def predict_from_features_workflow_name(self):
        return f"{self.name}.predict_from_features"

    @classmethod
    def _set_default(cls, fn=None, *, name):
        if fn is None:
            return partial(cls._set_default, name=name)

        setattr(cls, name, fn)
        return getattr(cls, name)

    def init(self, fn):
        self._init = fn
        return self._init

    def trainer(self, fn=None, /, **train_task_kwargs):
        if fn is None:
            return partial(self.trainer, **train_task_kwargs)
        self._trainer = fn
        self._trainer.__app_method__ = Endpoints.TRAINER
        self._train_task_kwargs = train_task_kwargs
        return self._trainer

    def predictor(self, fn=None, /, **predict_task_kwargs):
        if fn is None:
            return partial(self.trainer, **predict_task_kwargs)
        self._predictor = fn
        self._predictor.__app_method__ = Endpoints.PREDICTOR
        self._predict_task_kwargs = predict_task_kwargs
        return self._predictor
    
    def evaluator(self, fn):
        self._evaluator = fn
        return self._evaluator

    def train_workflow(self):
        dataset_task = self._dataset.dataset_task()
        train_task = self.train_task()

        [hyperparam_arg, hyperparam_type], *_ = train_task.python_interface.inputs.items()

        wf = Workflow(name=self.train_workflow_name)
        wf.add_workflow_input(hyperparam_arg, hyperparam_type)
        for arg, type in dataset_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        dataset_node = wf.add_entity(dataset_task, **{k: wf.inputs[k] for k in dataset_task.python_interface.inputs})
        train_node = wf.add_entity(train_task, **{hyperparam_arg: wf.inputs[hyperparam_arg], **dataset_node.outputs})
        wf.add_workflow_output("trained_model", train_node.outputs["trained_model"])
        wf.add_workflow_output("metrics", train_node.outputs["metrics"])
        return wf

    def predict_workflow(self):
        predict_task = self.predict_task()

        wf = Workflow(name=self.predict_workflow_name)
        for arg, type in predict_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        predict_node = wf.add_entity(predict_task, **{k: wf.inputs[k] for k in wf.inputs})
        for output_name, promise in predict_node.outputs.items():
            wf.add_workflow_output(output_name, promise)
        return wf

    def predict_from_features_workflow(self):
        predict_task = self.predict_from_features_task()

        wf = Workflow(name=self.predict_from_features_workflow_name)
        for arg, type in predict_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        predict_node = wf.add_entity(predict_task, **{k: wf.inputs[k] for k in wf.inputs})
        for output_name, promise in predict_node.outputs.items():
            wf.add_workflow_output(output_name, promise)
        return wf

    def train_task(self):
        if self._train_task:
            return self._train_task

        *_, hyperparameters_param = signature(self._init).parameters.values()
        *_, data_param = signature(self._trainer).parameters.values()

        init_kwargs = {}
        if self._init_cls:
            init_kwargs["init_cls"]= {
                "module": self._init_cls.__module__,
                "cls_name": self._init_cls.__name__,
            }

        @inner_task(
            fklearn_obj=self,
            input_parameters=OrderedDict([
                (p.name, p) for p in
                [hyperparameters_param, data_param.replace(name="train_data"), data_param.replace(name="test_data")]
            ]),
            return_annotation=NamedTuple(
                "TrainingResults",
                trained_model=signature(self._trainer).return_annotation,
                metrics=Dict[str, signature(self._evaluator).return_annotation]
            ),
            **({} if self._train_task_kwargs is None else self._train_task_kwargs),
        )
        def train_task(hyperparameters, train_data, test_data):
            trained_model = self._trainer(
                model=self._init(hyperparameters=hyperparameters, **init_kwargs), data=train_data
            )
            metrics = {
                "train": self._evaluator(model=trained_model, data=train_data),
                "test": self._evaluator(model=trained_model, data=test_data),
            }
            return trained_model, metrics

        self._train_task = train_task
        return train_task

    def predict_task(self):
        if self._predict_task:
            return self._predict_task

        predictor_sig = signature(self._predictor)
        model_param, *_ = predictor_sig.parameters.values()

        @inner_task(
            fklearn_obj=self,
            input_parameters=OrderedDict([
                (p.name, p) for p in
                [model_param, *signature(self._dataset._reader).parameters.values()]
            ]),
            return_annotation=predictor_sig.return_annotation,
        )
        def predict_task(model, **kwargs):
            data = self._dataset._reader(**kwargs)
            parsed_data = self._dataset._parser(data, **self._dataset.parser_kwargs)
            return self._predictor(model, self._dataset._unpack_features(parsed_data))

        self._predict_task = predict_task
        return predict_task

    def predict_from_features_task(self):
        if self._predict_from_features_task:
            return self._predict_from_features_task

        predictor_sig = signature(self._predictor)

        @inner_task(
            fklearn_obj=self,
            input_parameters=predictor_sig.parameters,
            return_annotation=predictor_sig.return_annotation,
        )
        def predict_from_features_task(model, features):
            parsed_data = self._dataset._parser(features, **self._dataset.parser_kwargs)
            return self._predictor(model, self._dataset._unpack_features(parsed_data))

        self._predict_from_features_task = predict_from_features_task
        return predict_from_features_task

    def train(self, hyperparameters: Dict[str, Any] = None, *, data: Any = None, lazy=False, **reader_kwargs):
        train_wf = self.train_workflow()
        if lazy:
            return train_wf
        if hyperparameters is None:
            raise ValueError("hyperparameters need to be provided when eager=True")

        trained_model, metrics = train_wf(hyperparameters=hyperparameters, **reader_kwargs)
        self._latest_model = trained_model
        self._latest_metrics = metrics
        return trained_model, metrics

    def predict(self, model: FlytePickle = None, features: Any = None, lazy=False, **reader_kwargs):
        # TODO: make a workflow constructor that predicts with feature literals
        if features is None:
            predict_wf = self.predict_workflow()
        else:
            predict_wf = self.predict_from_features_workflow()
        if lazy:
            return predict_wf

        if features is None:
            return predict_wf(model=model, **reader_kwargs)
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
def _default_init_model(self, init_cls: dict, hyperparameters: dict) -> FlytePickle:
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
            inputs: Dict = Body(...),
        ):
            if issubclass(type(inputs), BaseModel):
                inputs = inputs.dict()

            if not local:
                # TODO: make the model name a property of the Model object
                train_wf = model._remote.fetch_workflow(
                    name=f"{model_name}.train" if model_name else model.train_workflow_name
                )
                execution = model._remote.execute(train_wf, inputs=inputs, wait=True)
                trained_model = execution.outputs["trained_model"]
                metrics = execution.outputs["metrics"]
            else:
                trained_model, metrics = model.train(**inputs)
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
            inputs: Optional[Dict] = Body(None),
            features: Optional[List[Dict[str, Any]]] = Body(None)
        ):
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
                latest_model = latest_training_execution.outputs["trained_model"]
            else:
                if model._latest_model is None:
                    raise HTTPException(status_code=500, detail="trained model not found")
                latest_model = model._latest_model

            workflow_inputs = {"model": latest_model}
            if inputs:
                workflow_inputs.update(inputs)
                predict_wf = model._remote.fetch_workflow(
                    name=f"{model_name}.predict_workflow_name" if model_name else model.predict_workflow_name,
                    version=version,
                )
            elif features:
                features = model._dataset(features=features)()
                workflow_inputs["features"] = features
                predict_wf = model._remote.fetch_workflow(
                    name=(
                        f"{model_name}.predict_from_features_workflow_name"
                        if model_name
                        else model.predict_from_features_workflow_name
                    ),
                    version=version,
                )

            if not local:
                predictions = model._remote.execute(predict_wf, inputs=workflow_inputs, wait=True).outputs["o0"]
            else:
                predictions = model.predict(**workflow_inputs)
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
