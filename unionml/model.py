"""Model class for defining training, evaluation, and prediction."""

import inspect
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass, make_dataclass
from functools import partial
from inspect import Parameter, signature
from typing import IO, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import joblib
import sklearn
from dataclasses_json import dataclass_json
from flytekit import Workflow
from flytekit.configuration import Config
from flytekit.core.tracker import TrackedInstance
from flytekit.remote import FlyteRemote

from unionml.dataset import Dataset
from unionml.utils import inner_task


@dataclass
class BaseHyperparameters:
    """Hyperparameter base class"""

    pass


class ModelArtifact(NamedTuple):
    """Model artifact, containing a specific model object and optional metrics associated with it."""

    object: Any
    hyperparameters: Optional[Union[BaseHyperparameters, dict]] = None
    metrics: Optional[Dict[str, float]] = None


class Model(TrackedInstance):
    def __init__(
        self,
        name: str = "model",
        init: Union[Type, Callable] = None,
        *,
        dataset: Dataset,
        hyperparameter_config: Optional[Dict[str, Type]] = None,
    ):
        super().__init__()
        self.name = name
        self._init_callable = init
        self._hyperparameter_config = hyperparameter_config
        self._dataset = dataset
        self._artifact: Optional[ModelArtifact] = None

        # default component functions
        self._init = self._default_init
        self._saver = self._default_saver
        self._loader = self._default_loader

        # properties needed for deployment
        self._remote: Optional[FlyteRemote] = None
        self._image_name: Optional[str] = None
        self._config_file_path: Optional[str] = None
        self._registry: Optional[str] = None
        self._dockerfile: Optional[str] = None

        if self._dataset.name is None:
            self._dataset.name = f"{self.name}.dataset"

        # unionml-compiled tasks
        self._train_task = None
        self._predict_task = None
        self._predict_from_features_task = None

        # user-provided task kwargs
        self._train_task_kwargs = None
        self._predict_task_kwargs = None

        # dynamically defined types
        self._hyperparameter_type: Optional[Type] = None

    @property
    def artifact(self) -> Optional[ModelArtifact]:
        return self._artifact

    @artifact.setter
    def artifact(self, new_value: ModelArtifact):
        self._artifact = new_value

    @property
    def hyperparameter_type(self) -> Type:
        if self._hyperparameter_type is not None:
            return self._hyperparameter_type

        hyperparameter_fields: List[Any] = []
        if self._hyperparameter_config is None:
            # extract types from the init callable that instantiates a new model
            model_obj_sig = signature(self._init_callable)  # type: ignore

            # if any of the arguments are not type-annotated, default to using an untyped dictionary
            if any(p.annotation is inspect._empty for p in model_obj_sig.parameters.values()):
                return dict

            for hparam_name, hparam in model_obj_sig.parameters.items():
                hyperparameter_fields.append((hparam_name, hparam.annotation, field(default=hparam.default)))
        else:
            # extract types from hyperparameters Model init argument
            for hparam_name, hparam_type in self._hyperparameter_config.items():
                hyperparameter_fields.append((hparam_name, hparam_type))

        self._hyperparameter_type = dataclass_json(
            make_dataclass("Hyperparameters", hyperparameter_fields, bases=(BaseHyperparameters,))
        )
        return self._hyperparameter_type

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

    @property
    def trainer_params(self):
        return {
            name: param
            for name, param in signature(self._trainer).parameters.items()
            if param.kind == Parameter.KEYWORD_ONLY
        }

    def train_workflow(self):
        dataset_task = self._dataset.dataset_task()
        train_task = self.train_task()

        [
            hyperparam_arg,
            hyperparam_type,
        ], *_ = train_task.python_interface.inputs.items()

        wf = Workflow(name=self.train_workflow_name)

        # add hyperparameter argument
        wf.add_workflow_input(hyperparam_arg, hyperparam_type)

        # add dataset.reader arguments
        for arg, type in dataset_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        # add training keyword-only arguments
        trainer_param_types = {k: v.annotation for k, v in self.trainer_params.items()}
        for arg, type in trainer_param_types.items():
            wf.add_workflow_input(arg, type)

        dataset_node = wf.add_entity(
            dataset_task,
            **{k: wf.inputs[k] for k in dataset_task.python_interface.inputs},
        )
        train_node = wf.add_entity(
            train_task,
            **{
                hyperparam_arg: wf.inputs[hyperparam_arg],
                **dataset_node.outputs,
                **{arg: wf.inputs[arg] for arg in trainer_param_types},
            },
        )
        wf.add_workflow_output("trained_model", train_node.outputs["trained_model"])
        wf.add_workflow_output("hyperparameters", train_node.outputs["hyperparameters"])
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

        # make sure hyperparameter type signature is correct
        *_, hyperparameters_param = signature(self._init).parameters.values()
        hyperparameters_param = hyperparameters_param.replace(annotation=self.hyperparameter_type)

        # assume that reader_return_type is a dict with only a single entry
        [(data_arg_name, data_arg_type)] = self._dataset.reader_return_type.items()

        # get keyword-only training args
        @inner_task(
            unionml_obj=self,
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
                        *self.trainer_params.values(),
                    ]
                ]
            ),
            return_annotation=NamedTuple(
                "TrainingResults",
                trained_model=signature(self._trainer).return_annotation,
                hyperparameters=self.hyperparameter_type,
                metrics=Dict[str, signature(self._evaluator).return_annotation],
            ),
            **({} if self._train_task_kwargs is None else self._train_task_kwargs),
        )
        def train_task(**kwargs):
            hyperparameters = kwargs["hyperparameters"]
            raw_data = kwargs[data_arg_name]
            trainer_kwargs = {p: kwargs[p] for p in self.trainer_params}

            hyperparameters_dict = asdict(hyperparameters) if is_dataclass(hyperparameters) else hyperparameters
            training_data = self._dataset.get_data(raw_data)
            trained_model = self._trainer(
                self._init(hyperparameters=hyperparameters_dict),
                *training_data["train"],
                **trainer_kwargs,
            )
            metrics = {
                split_key: self._evaluator(trained_model, *training_data[split_key]) for split_key in ["train", "test"]
            }
            return trained_model, hyperparameters, metrics

        self._train_task = train_task
        return train_task

    def predict_task(self):
        if self._predict_task:
            return self._predict_task

        predictor_sig = signature(self._predictor)
        model_param, *_ = predictor_sig.parameters.values()
        model_param = model_param.replace(name="model")

        # assume that reader_return_type is a dict with only a single entry
        [(data_arg_name, data_arg_type)] = self._dataset.reader_return_type.items()
        data_param = Parameter(data_arg_name, kind=Parameter.KEYWORD_ONLY, annotation=data_arg_type)

        # TODO: make sure return type is not None
        @inner_task(
            unionml_obj=self,
            input_parameters=OrderedDict([(p.name, p) for p in [model_param, data_param]]),
            return_annotation=predictor_sig.return_annotation,
            **self._predict_task_kwargs,
        )
        def predict_task(model, **kwargs):
            parsed_data = self._dataset._parser(kwargs[data_arg_name], **self._dataset.parser_kwargs)
            return self._predictor(model, self._dataset._feature_processor(parsed_data))

        self._predict_task = predict_task
        return predict_task

    # TODO: see if this can be merged with predict_task
    def predict_from_features_task(self):
        if self._predict_from_features_task:
            return self._predict_from_features_task

        predictor_sig = signature(self._predictor)
        model_param, *_ = predictor_sig.parameters.values()
        model_param = model_param.replace(name="model")

        # assume that reader_return_type is a dict with only a single entry
        [(_, data_arg_type)] = self._dataset.reader_return_type.items()
        data_param = Parameter("features", kind=Parameter.KEYWORD_ONLY, annotation=data_arg_type)

        @inner_task(
            unionml_obj=self,
            input_parameters=OrderedDict([("model", model_param), ("features", data_param)]),
            return_annotation=predictor_sig.return_annotation,
            **self._predict_task_kwargs,
        )
        def predict_from_features_task(model, features):
            parsed_data = self._dataset._parser(features, **self._dataset.parser_kwargs)
            return self._predictor(model, self._dataset._feature_processor(parsed_data))

        self._predict_from_features_task = predict_from_features_task
        return predict_from_features_task

    def train(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **reader_kwargs,
    ) -> Tuple[Any, Any]:
        trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs
        model_obj, hyperparameters, metrics = self.train_workflow()(
            hyperparameters=self.hyperparameter_type(**({} if hyperparameters is None else hyperparameters)),
            **{**reader_kwargs, **trainer_kwargs},
        )
        self.artifact = ModelArtifact(model_obj, hyperparameters, metrics)
        return model_obj, metrics

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
        return self._saver(self.artifact.object, self.artifact.hyperparameters, file, *args, **kwargs)

    def load(self, file, *args, **kwargs):
        return self._loader(file, *args, **kwargs)

    def serve(self, app):
        """Create a FastAPI serving app."""
        from unionml.fastapi import serving_app

        serving_app(self, app)

    def remote(
        self,
        registry: Optional[str] = None,
        image_name: str = None,
        dockerfile: str = "Dockerfile",
        config_file_path: Optional[str] = None,
        project: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        self._config_file_path = config_file_path
        self._registry = registry
        self._image_name = image_name
        self._dockerfile = dockerfile
        self._remote = FlyteRemote(
            config=Config.auto(config_file=self._config_file_path),
            default_project=project,
            default_domain=domain,
        )

    def remote_deploy(self):
        """Deploy model services to a Flyte backend."""
        from unionml import remote

        version = remote.get_app_version()
        image = remote.get_image_fqn(self, version, self._image_name)

        # FlyteRemote needs to be re-instantiated after setting this environment variable so that the workflow's
        # default image is set correctly. This can be simplified after flytekit config improvements
        # are merged: https://github.com/flyteorg/flytekit/pull/857
        os.environ["FLYTE_INTERNAL_IMAGE"] = image or ""
        self._remote = FlyteRemote(
            config=Config.auto(config_file=self._config_file_path),
            default_project=self._remote._default_project,
            default_domain=self._remote._default_domain,
        )

        remote.create_project(self._remote, self._remote._default_project)
        if self._remote.config.platform.endpoint.startswith("localhost"):
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
            remote.deploy_wf(wf, self._remote, image, *args)

    def remote_train(
        self,
        app_version: str = None,
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **reader_kwargs,
    ) -> ModelArtifact:
        if self._remote is None:
            raise RuntimeError("First configure the remote client with the `Model.remote` method")
        train_wf = self._remote.fetch_workflow(name=self.train_workflow_name, version=app_version)
        execution = self._remote.execute(
            train_wf,
            inputs={
                "hyperparameters": self.hyperparameter_type(**({} if hyperparameters is None else hyperparameters)),
                **{**reader_kwargs, **trainer_kwargs},
            },
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

        from unionml import remote

        app_version = app_version or remote.get_app_version()
        model_artifact = remote.get_latest_model_artifact(self, app_version)

        if (features is not None and len(reader_kwargs) > 0) or (features is None and len(reader_kwargs) == 0):
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

    def _default_saver(
        self,
        model_obj: Any,
        hyperparameters: Union[dict, BaseHyperparameters],
        file: Union[str, os.PathLike, IO],
        *args,
        **kwargs,
    ) -> Any:
        import torch

        hyperparameters = asdict(hyperparameters) if is_dataclass(hyperparameters) else hyperparameters
        if isinstance(model_obj, sklearn.base.BaseEstimator):
            return joblib.dump({"model_state": model_obj, "hyperparameters": hyperparameters}, file, *args, **kwargs)
        elif isinstance(model_obj, torch.nn.Module):
            torch.save(
                {"model_state": model_obj.state_dict(), "hyperparameters": hyperparameters},
                file,
                *args,
                **kwargs,
            )
            return file

        raise NotImplementedError(
            f"Default saver not defined for type {type(model_obj)}. Use the Model.saver decorator to define one."
        )

    def _default_loader(self, file: Union[str, os.PathLike, IO], *args, **kwargs) -> Any:
        import torch

        init = self._init_callable if self._init == self._default_init else self._init or self._init_callable
        model_type = init if inspect.isclass(init) else signature(init).return_annotation if init is not None else init

        if issubclass(model_type, sklearn.base.BaseEstimator):
            deserialized_model = joblib.load(file, *args, **kwargs)
            return deserialized_model["model_state"]
        elif issubclass(model_type, torch.nn.Module):
            deserialized_model = torch.load(file, *args, **kwargs)
            model = model_type(**deserialized_model["hyperparameters"])
            model.load_state_dict(deserialized_model["model_state"])
            return model

        raise NotImplementedError(
            f"Default loader not defined for type {model_type}. Use the Model.loader decorator to define one."
        )
