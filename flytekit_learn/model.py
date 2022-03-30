"""Model class for defining training, evaluation, and prediction."""

import inspect
import os
from collections import OrderedDict
from functools import partial
from inspect import Parameter, signature
from typing import IO, Any, Callable, Dict, NamedTuple, Optional, Type, Union

import joblib
import sklearn
from flytekit import Workflow
from flytekit.core.tracker import TrackedInstance
from flytekit.remote import FlyteRemote

from flytekit_learn.dataset import Dataset
from flytekit_learn.utils import inner_task


class ModelArtifact(NamedTuple):
    object: Any
    metrics: Optional[Dict[str, float]] = None


class Model(TrackedInstance):
    def __init__(
        self,
        name: str = "model",
        init: Union[Type, Callable] = None,
        *,
        dataset: Dataset,
        hyperparameters: Optional[Dict[str, Type]] = None,
    ):
        super().__init__()
        self.name = name
        self._init_callable = init
        self._hyperparameters = hyperparameters
        self._dataset = dataset
        self._artifact: Optional[ModelArtifact] = None

        # default component functions
        self._init = self._default_init
        self._saver = self._default_saver
        self._loader = self._default_loader

        # properties needed for deployment
        self._remote: Optional[FlyteRemote] = None
        self._config_file_path: Optional[str] = None
        self._registry: Optional[str] = None
        self._dockerfile: Optional[str] = None

        if self._dataset.name is None:
            self._dataset.name = f"{self.name}.dataset"

        # fklearn-compiled tasks
        self._train_task = None
        self._predict_task = None
        self._predict_from_features_task = None

        # user-provided task kwargs
        self._train_task_kwargs = None
        self._predict_task_kwargs = None

    @property
    def artifact(self) -> Optional[ModelArtifact]:
        return self._artifact

    @artifact.setter
    def artifact(self, new_value: ModelArtifact):
        self._artifact = new_value

    @property
    def config_file_path(self) -> Optional[str]:
        return self._config_file_path

    @property
    def registry(self) -> Optional[str]:
        return self._registry

    @property
    def dockerfile(self) -> Optional[str]:
        return self._dockerfile

    @property
    def train_workflow_name(self):
        return f"{self.name}.train"

    @property
    def predict_workflow_name(self):
        return f"{self.name}.predict"

    @property
    def predict_from_features_workflow_name(self):
        return f"{self.name}.predict_from_features"

    def init(self, fn):
        self._init = fn
        return self._init

    def trainer(self, fn=None, **train_task_kwargs):
        if fn is None:
            return partial(self.trainer, **train_task_kwargs)
        self._trainer = fn
        self._train_task_kwargs = train_task_kwargs
        return self._trainer

    def predictor(self, fn=None, **predict_task_kwargs):
        if fn is None:
            return partial(self.trainer, **predict_task_kwargs)
        self._predictor = fn
        self._predict_task_kwargs = predict_task_kwargs
        return self._predictor

    def evaluator(self, fn):
        self._evaluator = fn
        return self._evaluator

    def saver(self, fn):
        self._saver = fn
        return self._saver

    def loader(self, fn):
        self._loader = fn
        return self._loader

    def train_workflow(self):
        dataset_task = self._dataset.dataset_task()
        train_task = self.train_task()

        [
            hyperparam_arg,
            hyperparam_type,
        ], *_ = train_task.python_interface.inputs.items()

        wf = Workflow(name=self.train_workflow_name)
        wf.add_workflow_input(hyperparam_arg, hyperparam_type)
        for arg, type in dataset_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        dataset_node = wf.add_entity(
            dataset_task,
            **{k: wf.inputs[k] for k in dataset_task.python_interface.inputs},
        )
        train_node = wf.add_entity(
            train_task,
            **{hyperparam_arg: wf.inputs[hyperparam_arg], **dataset_node.outputs},
        )
        wf.add_workflow_output("trained_model", train_node.outputs["trained_model"])
        wf.add_workflow_output("metrics", train_node.outputs["metrics"])
        return wf

    def predict_workflow(self):
        dataset_task = self._dataset.dataset_task()
        predict_task = self.predict_task()

        wf = Workflow(name=self.predict_workflow_name)
        model_arg_name, *_ = predict_task.python_interface.inputs.keys()
        wf.add_workflow_input("model", predict_task.python_interface.inputs[model_arg_name])
        for arg, type in dataset_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        dataset_node = wf.add_entity(
            dataset_task,
            **{k: wf.inputs[k] for k in dataset_task.python_interface.inputs},
        )
        predict_node = wf.add_entity(predict_task, **{"model": wf.inputs["model"], **dataset_node.outputs})
        for output_name, promise in predict_node.outputs.items():
            wf.add_workflow_output(output_name, promise)
        return wf

    def predict_from_features_workflow(self):
        predict_task = self.predict_from_features_task()

        wf = Workflow(name=self.predict_from_features_workflow_name)
        for i, (arg, type) in enumerate(predict_task.python_interface.inputs.items()):
            # assume that the first argument is the model object
            wf.add_workflow_input("model" if i == 0 else arg, type)

        predict_node = wf.add_entity(predict_task, **{k: wf.inputs[k] for k in wf.inputs})
        for output_name, promise in predict_node.outputs.items():
            wf.add_workflow_output(output_name, promise)
        return wf

    def train_task(self):
        if self._train_task:
            return self._train_task

        *_, hyperparameters_param = signature(self._init).parameters.values()

        # assume that reader_return_type is a dict with only a single entry
        [(data_arg_name, data_arg_type)] = self._dataset.reader_return_type.items()

        # TODO: make sure return type is not None
        @inner_task(
            fklearn_obj=self,
            input_parameters=OrderedDict(
                [
                    (p.name, p)
                    for p in [
                        hyperparameters_param,
                        Parameter(
                            data_arg_name,
                            kind=Parameter.KEYWORD_ONLY,
                            annotation=data_arg_type,
                        ),
                    ]
                ]
            ),
            return_annotation=NamedTuple(
                "TrainingResults",
                trained_model=signature(self._trainer).return_annotation,
                metrics=Dict[str, signature(self._evaluator).return_annotation],
            ),
            **({} if self._train_task_kwargs is None else self._train_task_kwargs),
        )
        def train_task(**kwargs):
            hyperparameters = kwargs["hyperparameters"]
            raw_data = kwargs[data_arg_name]
            training_data = self._dataset.get_data(raw_data)
            trained_model = self._trainer(
                self._init(hyperparameters=hyperparameters),
                *training_data["train"],
            )
            metrics = {
                split_key: self._evaluator(trained_model, *training_data[split_key]) for split_key in ["train", "test"]
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
            **self._predict_task_kwargs,
        )
        def predict_task(model, **kwargs):
            parsed_data = self._dataset._parser(kwargs[data_arg_name], **self._dataset.parser_kwargs)
            return self._predictor(model, self._dataset._feature_getter(parsed_data))

        self._predict_task = predict_task
        return predict_task

    def predict_from_features_task(self):
        if self._predict_from_features_task:
            return self._predict_from_features_task

        predictor_sig = signature(self._predictor)

        @inner_task(
            fklearn_obj=self,
            input_parameters=OrderedDict(
                [
                    # assume that the first argument of the predictor represents the model object
                    ("model", p.replace(name="model")) if i == 0 else (name, p)
                    for i, (name, p) in enumerate(predictor_sig.parameters.items())
                ]
            ),
            return_annotation=predictor_sig.return_annotation,
            **self._predict_task_kwargs,
        )
        def predict_from_features_task(model, features):
            parsed_data = self._dataset._parser(features, **self._dataset.parser_kwargs)
            return self._predictor(model, self._dataset._feature_getter(parsed_data))

        self._predict_from_features_task = predict_from_features_task
        return predict_from_features_task

    def train(
        self,
        hyperparameters: Dict[str, Any] = None,
        **reader_kwargs,
    ) -> ModelArtifact:
        model_obj, metrics = self.train_workflow()(
            hyperparameters={} if hyperparameters is None else hyperparameters,
            **reader_kwargs,
        )
        self.artifact = ModelArtifact(model_obj, metrics)
        return self.artifact

    def predict(
        self,
        features: Any = None,
        **reader_kwargs,
    ):
        if self.artifact is None:
            raise RuntimeError(
                "ModelArtifact not found. You must train a model first with the `train` method before generating "
                "predictions."
            )
        if features is None:
            return self.predict_workflow()(model=self.artifact.object, **reader_kwargs)
        return self.predict_from_features_workflow()(model=self.artifact.object, features=features)

    def save(self, file, *args, **kwargs):
        if self.artifact is None:
            raise AttributeError("`artifact` property is None. Call the `train` method to train a model first")
        return self._saver(self.artifact.object, file, *args, **kwargs)

    def load(self, file, *args, **kwargs):
        return self._loader(file, *args, **kwargs)

    def serve(self, app):
        """Create a FastAPI serving app."""
        from flytekit_learn.fastapi import serving_app

        serving_app(self, app)

    def remote(
        self,
        registry: Optional[str] = None,
        dockerfile: str = "Dockerfile",
        config_file_path: Optional[str] = None,
        project: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        self._config_file_path = config_file_path
        self._registry = registry
        self._dockerfile = dockerfile
        self._remote = FlyteRemote.from_config(
            config_file_path=config_file_path,
            default_project=project,
            default_domain=domain,
        )

    def remote_deploy(self):
        """Deploy model services to a Flyte backend."""
        from flytekit_learn import remote

        version = remote.get_app_version()
        image = remote.get_image_fqn(self, version)

        # FlyteRemote needs to be re-instantiated after setting this environment variable so that the workflow's
        # default image is set correctly. This can be simplified after flytekit config improvements
        # are merged: https://github.com/flyteorg/flytekit/pull/857
        os.environ["FLYTE_INTERNAL_IMAGE"] = image or ""
        self._remote = FlyteRemote.from_config(
            config_file_path=self._config_file_path,
            default_project=self._remote._default_project,
            default_domain=self._remote._default_domain,
        )

        remote.create_project(self._remote, self._remote._default_project)
        if self._remote._flyte_admin_url.startswith("localhost"):
            # assume that a localhost flyte_admin_url means that we want to use Flyte sandbox
            remote.sandbox_docker_build(self, image)
        else:
            remote.docker_build_push(self, image)

        args = [self._remote._default_project, self._remote._default_domain, version]
        for wf in [
            self.train_workflow(),
            self.predict_workflow(),
            self.predict_from_features_workflow(),
        ]:
            remote.deploy_wf(wf, self._remote, *args)

    def remote_train(
        self,
        app_version: str = None,
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
        **reader_kwargs,
    ) -> ModelArtifact:
        if self._remote is None:
            raise RuntimeError("First configure the remote client with the `Model.remote` method")
        train_wf = self._remote.fetch_workflow(name=self.train_workflow_name, version=app_version)
        execution = self._remote.execute(
            train_wf,
            inputs={"hyperparameters": {} if hyperparameters is None else hyperparameters, **reader_kwargs},
            wait=True,
        )
        return ModelArtifact(
            execution.outputs["trained_model"],
            execution.outputs["metrics"],
        )

    def remote_predict(
        self,
        app_version: str = None,
        *,
        features: Any = None,
        **reader_kwargs,
    ):
        if self._remote is None:
            raise RuntimeError("First configure the remote client with the `Model.remote` method")

        from flytekit_learn import remote

        app_version = app_version or remote.get_app_version()
        model_artifact = remote.get_latest_model_artifact(self, app_version)

        if (features is not None and reader_kwargs is not None) or (features and reader_kwargs):
            raise ValueError("You must provide only one of `features` or `reader_kwargs`")

        inputs = {"model": model_artifact.object}
        if features is None:
            workflow_name = self.predict_workflow_name
            inputs.update(reader_kwargs)
        else:
            workflow_name = self.predict_from_features_workflow_name
            inputs.update({"features": features})

        predict_wf = self._remote.fetch_workflow(
            self._remote._default_project,
            self._remote._default_domain,
            workflow_name,
            app_version,
        )
        execution = self._remote.execute(predict_wf, inputs=inputs, wait=True)
        predictions, *_ = execution.outputs.values()
        return predictions

    def _default_init(self, hyperparameters: dict) -> Any:
        if self._init_callable is None:
            raise ValueError(
                "When using the _default_init method, you must specify the init argument to the Model constructor."
            )
        return self._init_callable(**hyperparameters)

    def _default_saver(self, model_obj: Any, file: Union[str, os.PathLike, IO], *args, **kwargs) -> Any:
        if isinstance(model_obj, sklearn.base.BaseEstimator):
            return joblib.dump(model_obj, file, *args, **kwargs)

        raise NotImplementedError(
            f"Default saver not defined for type {type(model_obj)}. Use the Model.saver decorator to define one."
        )

    def _default_loader(self, file: Union[str, os.PathLike, IO], *args, **kwargs) -> Any:
        init = self._init_callable if self._init == self._default_init else self._init or self._init_callable
        model_type = init if inspect.isclass(init) else signature(init).return_annotation if init is not None else init

        if issubclass(model_type, sklearn.base.BaseEstimator):
            return joblib.load(file, *args, **kwargs)

        raise NotImplementedError(
            f"Default loader not defined for type {model_type}. Use the Model.loader decorator to define one."
        )
