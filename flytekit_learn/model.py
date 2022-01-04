"""Model class for defining training, evaluation, and prediction."""

import importlib
from collections import OrderedDict
from functools import partial, wraps
from inspect import signature, Parameter
from typing import Any, Callable, Dict, Optional, NamedTuple, Type, Union

from flytekit import Workflow
from flytekit.core.tracker import TrackedInstance
from flytekit.remote import FlyteRemote
from flytekit.types.pickle import FlytePickle

from flytekit_learn.dataset import Dataset
from flytekit_learn.fastapi import app_wrapper
from flytekit_learn.utils import inner_task


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
        self._train_task_kwargs = train_task_kwargs
        return self._trainer

    def predictor(self, fn=None, /, **predict_task_kwargs):
        if fn is None:
            return partial(self.trainer, **predict_task_kwargs)
        self._predictor = fn
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
        dataset_task = self._dataset.dataset_task()
        predict_task = self.predict_task()

        wf = Workflow(name=self.predict_workflow_name)
        wf.add_workflow_input("model", predict_task.python_interface.inputs["model"])
        for arg, type in dataset_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        dataset_node = wf.add_entity(dataset_task, **{k: wf.inputs[k] for k in dataset_task.python_interface.inputs})
        predict_node = wf.add_entity(predict_task, **{"model": wf.inputs["model"], **dataset_node.outputs})
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

        init_kwargs = {}
        if self._init_cls:
            init_kwargs["init_cls"]= {
                "module": self._init_cls.__module__,
                "cls_name": self._init_cls.__name__,
            }

        # assume that reader_return_type is a dict with only a single entry
        [(data_arg_name, data_arg_type)] = self._dataset.reader_return_type.items()

        # TODO: make sure return type is not None
        @inner_task(
            fklearn_obj=self,
            input_parameters=OrderedDict([
                (p.name, p) for p in
                [
                    hyperparameters_param,
                    Parameter(data_arg_name, kind=Parameter.KEYWORD_ONLY, annotation=data_arg_type)
                ]
            ]),
            return_annotation=NamedTuple(
                "TrainingResults",
                trained_model=signature(self._trainer).return_annotation,
                metrics=Dict[str, signature(self._evaluator).return_annotation]
            ),
            **({} if self._train_task_kwargs is None else self._train_task_kwargs),
        )
        def train_task(**kwargs):
            hyperparameters = kwargs["hyperparameters"]
            data = kwargs[data_arg_name]
            train_split, test_split = self._dataset._splitter(data=data, **self._dataset.splitter_kwargs)
            train_data = self._dataset._parser(train_split, **self._dataset.parser_kwargs)
            test_data = self._dataset._parser(test_split, **self._dataset.parser_kwargs)
            trained_model = self._trainer(self._init(hyperparameters=hyperparameters, **init_kwargs), *train_data)
            metrics = {
                "train": self._evaluator(trained_model, *train_data),
                "test": self._evaluator(trained_model, *test_data),
            }
            return trained_model, metrics

        self._train_task = train_task
        return train_task

    def predict_task(self):
        if self._predict_task:
            return self._predict_task

        predictor_sig = signature(self._predictor)
        model_param, *_ = predictor_sig.parameters.values()

        # assume that reader_return_type is a dict with only a single entry
        [(data_arg_name, data_arg_type)] = self._dataset.reader_return_type.items()
        data_param = Parameter(data_arg_name, kind=Parameter.KEYWORD_ONLY, annotation=data_arg_type)

        # TODO: make sure return type is not None
        @inner_task(
            fklearn_obj=self,
            input_parameters=OrderedDict([(p.name, p) for p in [model_param, data_param]]),
            return_annotation=predictor_sig.return_annotation,
        )
        def predict_task(model, **kwargs):
            parsed_data = self._dataset._parser(kwargs[data_arg_name], **self._dataset.parser_kwargs)
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

    def serve(
        self,
        app,
        default_endpoints: bool = True,
        train_endpoint: str = "/train",
        predict_endpoint: str = "/predict"
    ):
        app_wrapper(self, app, default_endpoints, train_endpoint=train_endpoint, predict_endpoint=predict_endpoint)


@Model._set_default(name="_init")
def _default_init_model(self, init_cls: dict, hyperparameters: dict) -> FlytePickle:
    module = importlib.import_module(init_cls["module"])
    cls = getattr(module, init_cls["cls_name"])
    return cls(**hyperparameters)
