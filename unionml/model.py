"""Model class for defining training, evaluation, and prediction."""

import inspect
import os
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass, make_dataclass
from functools import partial
from inspect import Parameter, signature
from typing import IO, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import joblib
import pandas as pd
import sklearn
from dataclasses_json import dataclass_json
from fastapi import FastAPI
from flytekit import Workflow
from flytekit.configuration import Config
from flytekit.core.tracker import TrackedInstance
from flytekit.remote import FlyteRemote
from flytekit.remote.executions import FlyteWorkflowExecution
from flytekit.types.pickle import FlytePickle

import unionml.type_guards as type_guards
from unionml._logging import logger
from unionml.dataset import Dataset
from unionml.defaults import DEFAULT_RESOURCES
from unionml.utils import inner_task, is_keras_model, is_pytorch_model


@dataclass
class BaseHyperparameters:
    """Hyperparameter base class

    This class is used to auto-generate the hyperparameter type based on the ``hyperparameter_config`` argument
    or ``init`` callable signature in the :py:class:`unionml.model.Model`.
    """

    pass


class ModelArtifact(NamedTuple):
    """Model artifact, containing a specific model object and optional metrics associated with it."""

    #: model object
    model_object: Any

    #: hyperparameters associated with the model object
    hyperparameters: Optional[Union[BaseHyperparameters, dict]] = None

    #: metrics associated with the model object
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
        """Initialize a UnionML Model.

        The term *UnionML Model*  refers to the specification of a model, which the user defines through
        the functional entrypoints, e.g. :meth:`unionml.model.Model.trainer`. The term *model object* is used to refer
        to some instance of model from a machine learning framework such as the subclasses of the ``BaseEstimator``
        class in sklearn, ``Module`` in pytorch, etc.

        :param name: name of the model
        :param init: a class or callable that produces a model object (e.g. an sklearn estimator) when invoked.
        :param dataset: a UnionML Dataset object to bind to the model.
        :param hyperparameter_config: A dictionary mapping hyperparameter names to types. This is used to
            determine the hyperparameter names and types associated with the model object produced by
            the ``init`` argument. For example:

            >>> {
            ...    "hyperparameter1": int,
            ...    "hyperparameter2": str,
            ...    "hyperparameter3": float,
            ... }
        """
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
        self._image_name: Optional[str] = None
        self._config_file: Optional[str] = None
        self._registry: Optional[str] = None
        self._dockerfile: Optional[str] = None
        self._project: Optional[str] = None
        self._domain: Optional[str] = None
        self.__remote__: Optional[FlyteRemote] = None

        if self._dataset.name is None:
            self._dataset.name = f"{self.name}.dataset"

        # unionml-compiled tasks
        self._train_task = None
        self._predict_task = None
        self._predict_from_features_task = None

        # user-provided task kwargs
        self._train_task_kwargs: Optional[Dict[str, Any]] = None
        self._predict_task_kwargs: Optional[Dict[str, Any]] = None

        # dynamically defined types
        self._hyperparameter_type: Optional[Type] = None

        self._patch_destination_dir: Optional[str] = None

    @property
    def artifact(self) -> Optional[ModelArtifact]:
        """Model artifact associated with the ``unionml.Model`` ."""
        return self._artifact

    @artifact.setter
    def artifact(self, new_value: ModelArtifact):
        self._artifact = new_value

    @property
    def dataset(self) -> Dataset:
        """Exposes the ``unionml.Dataset`` associated with the model."""
        return self._dataset

    @property
    def hyperparameter_type(self) -> Type:
        """Hyperparameter type of the model object based on the ``init`` function signature."""
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
    def config_file(self) -> Optional[str]:
        """Path to the config file associated with the Flyte backend."""
        return self._config_file

    @property
    def registry(self) -> Optional[str]:
        """Docker registry used to push UnionML app."""
        return self._registry

    @property
    def dockerfile(self) -> Optional[str]:
        """Path to Docker file used to package the UnionML app."""
        return self._dockerfile

    @property
    def train_workflow_name(self):
        """Name of the training workflow."""
        return f"{self.name}.train"

    @property
    def predict_workflow_name(self):
        """Name of the prediction workflow used to generate predictions from the ``dataset.reader`` ."""
        return f"{self.name}.predict"

    @property
    def predict_from_features_workflow_name(self):
        """Name of the prediction workflow used to generate predictions from raw features."""
        return f"{self.name}.predict_from_features"

    def init(self, fn):
        """Register a function for initializing a model object."""
        self._init = fn
        return self._init

    def trainer(self, fn: Callable = None, **train_task_kwargs):
        """Register a function for training a model object.

        This function is the primary entrypoint for defining your application's model-training behavior.

        See the :ref:`User Guide <model_trainer>` for more.

        :param fn: function to use as the trainer.
        :param train_task_kwargs: keyword arguments to pass into the
            :doc:`flytekit task <flytekit:generated/flytekit.task>` that will be composed of the input ``fn`` and
            functions defined in the bound :py:class:`~unionml.dataset.Dataset`.
        """
        if fn is None:
            return partial(self.trainer, **train_task_kwargs)

        if self.dataset._parser == self.dataset._default_parser:
            # Use the reader/loader datatype if parser is the default parser, otherwise use parser return type.
            # Additionally, special-case dataframes so that they are treated as two dataframes: one for features
            # and another for targets.
            expected_types = (
                tuple([self.dataset.dataset_datatype["data"]] * 2)
                if self.dataset.dataset_datatype["data"] is pd.DataFrame
                else (self.dataset.dataset_datatype["data"],)
            )
        else:
            expected_types = self.dataset.parser_return_types

        type_guards.guard_trainer(fn, self.model_type, expected_types)
        self._trainer = fn
        self._train_task_kwargs = {"requests": DEFAULT_RESOURCES, "limits": DEFAULT_RESOURCES, **train_task_kwargs}
        return self._trainer

    def predictor(self, fn=None, **predict_task_kwargs):
        """Register a function that generates predictions from a model object.

        This function is the primary entrypoint for defining your application's prediction behavior.

        See the :ref:`User Guide <model_predictor>` for more.

        :param fn: function to use as the predictor.
        :param train_task_kwargs: keyword arguments to pass into the
            :doc:`flytekit task <flytekit:generated/flytekit.task>` that will be composed of the input ``fn`` and
            functions defined in the bound :py:class:`~unionml.dataset.Dataset`.
        """
        if fn is None:
            return partial(self.predictor, **predict_task_kwargs)

        type_guards.guard_predictor(fn, self.model_type, self._dataset.feature_type)
        self._predictor = fn
        self._predict_task_kwargs = {
            "requests": DEFAULT_RESOURCES,
            "limits": DEFAULT_RESOURCES,
            **predict_task_kwargs,
        }
        return self._predictor

    def evaluator(self, fn):
        """Register a function for producing metrics for given model object."""

        if self.dataset._parser == self.dataset._default_parser:
            # Use the reader/loader datatype if parser is the default parser, otherwise use parser return type.
            # Additionally, special-case dataframes so that they are treated as two dataframes: one for features
            # and another for targets.
            expected_types = (
                tuple([self.dataset.dataset_datatype["data"]] * 2)
                if self.dataset.dataset_datatype["data"] is pd.DataFrame
                else (self.dataset.dataset_datatype["data"],)
            )
        else:
            expected_types = self.dataset.parser_return_types

        type_guards.guard_evaluator(fn, self.model_type, expected_types)
        self._evaluator = fn
        return self._evaluator

    def saver(self, fn):
        """Register a function for serializing a model object to disk."""
        self._saver = fn
        return self._saver

    def loader(self, fn):
        """Register a function for deserializing a model object to disk."""
        self._loader = fn
        return self._loader

    @property
    def trainer_params(self) -> Dict[str, Parameter]:
        """Parameters used to create a Flyte workflow for model object training."""
        return {
            name: param
            for name, param in signature(self._trainer).parameters.items()
            if param.kind == Parameter.KEYWORD_ONLY
        }

    def train_workflow(self):
        """Create a Flyte training workflow for model object training."""
        dataset_task = self._dataset.dataset_task()
        train_task = self.train_task()

        [
            hyperparam_arg,
            hyperparam_type,
        ], *_ = train_task.python_interface.inputs.items()

        wf = Workflow(name=self.train_workflow_name)

        # add hyperparameter argument
        wf.add_workflow_input(hyperparam_arg, hyperparam_type)

        # add loader, splitter, parser kwargs
        wf.add_workflow_input("loader_kwargs", self._dataset.loader_kwargs_type)
        wf.add_workflow_input("splitter_kwargs", self._dataset.splitter_kwargs_type)
        wf.add_workflow_input("parser_kwargs", self._dataset.parser_kwargs_type)

        # add dataset.reader arguments
        for arg, type in dataset_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        # add training kwargs
        trainer_param_types = {k: v.annotation for k, v in self.trainer_params.items()}
        for arg, type in trainer_param_types.items():
            wf.add_workflow_input(arg, type)

        dataset_node = wf.add_entity(
            dataset_task,
            **{k: wf.inputs[k] for k in dataset_task.python_interface.inputs},
        )
        (_, dataset_promise), *_ = dataset_node.outputs.items()
        train_node = wf.add_entity(
            train_task,
            **{
                hyperparam_arg: wf.inputs[hyperparam_arg],
                **{"data": dataset_promise},
                **{arg: wf.inputs[arg] for arg in trainer_param_types},
                **{arg: wf.inputs[arg] for arg in ["loader_kwargs", "splitter_kwargs", "parser_kwargs"]},
            },
        )
        wf.add_workflow_output("model_object", train_node.outputs["model_object"])
        wf.add_workflow_output("hyperparameters", train_node.outputs["hyperparameters"])
        wf.add_workflow_output("metrics", train_node.outputs["metrics"])
        return wf

    def predict_workflow(self):
        """Create a Flyte prediction workflow using features from the ``dataset.reader`` as the data source."""
        dataset_task = self._dataset.dataset_task()
        predict_task = self.predict_task()

        wf = Workflow(name=self.predict_workflow_name)
        model_arg_name, *_ = predict_task.python_interface.inputs.keys()
        wf.add_workflow_input("model_object", predict_task.python_interface.inputs[model_arg_name])
        for arg, type in dataset_task.python_interface.inputs.items():
            wf.add_workflow_input(arg, type)

        dataset_node = wf.add_entity(
            dataset_task,
            **{k: wf.inputs[k] for k in dataset_task.python_interface.inputs},
        )
        (_, dataset_promise), *_ = dataset_node.outputs.items()
        predict_node = wf.add_entity(
            predict_task, **{"model_object": wf.inputs["model_object"], **{"data": dataset_promise}}
        )
        for output_name, promise in predict_node.outputs.items():
            wf.add_workflow_output(output_name, promise)
        return wf

    def predict_from_features_workflow(self):
        """Create a Flyte prediction workflow using raw features."""
        predict_task = self.predict_from_features_task()

        wf = Workflow(name=self.predict_from_features_workflow_name)
        for i, (arg, type) in enumerate(predict_task.python_interface.inputs.items()):
            # assume that the first argument is the model object
            wf.add_workflow_input("model_object" if i == 0 else arg, type)

        predict_node = wf.add_entity(predict_task, **{k: wf.inputs[k] for k in wf.inputs})
        for output_name, promise in predict_node.outputs.items():
            wf.add_workflow_output(output_name, promise)
        return wf

    def train_task(self):
        """Create a Flyte task for training a model object.

        This is used in the Flyte workflow produced by ``train_workflow``.
        """
        if self._train_task:
            return self._train_task

        # make sure hyperparameter type signature is correct
        *_, hyperparameters_param = signature(self._init).parameters.values()
        hyperparameters_param = hyperparameters_param.replace(annotation=self.hyperparameter_type)

        # assume that dataset_datatype is a dict with only a single entry
        [(data_arg_name, data_arg_type)] = self._dataset.dataset_datatype.items()

        # get keyword-only training args
        @inner_task(
            unionml_obj=self,
            input_parameters=OrderedDict(
                [
                    (p.name, p)
                    for p in [
                        # hyperparameters
                        hyperparameters_param,
                        # raw data
                        Parameter(
                            data_arg_name,
                            kind=Parameter.KEYWORD_ONLY,
                            annotation=data_arg_type,
                        ),
                        # loader, splitter, and parser kwargs
                        *[
                            Parameter(arg, kind=Parameter.KEYWORD_ONLY, annotation=dict)
                            for arg in ["loader_kwargs", "splitter_kwargs", "parser_kwargs"]
                        ],
                        # trainer kwargs
                        *self.trainer_params.values(),
                    ]
                ]
            ),
            return_annotation=NamedTuple(
                "ModelArtifact",
                model_object=signature(self._trainer).return_annotation,
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
            model_object = self._trainer(
                self._init(hyperparameters=hyperparameters_dict),
                *training_data["train"],
                **trainer_kwargs,
            )
            metrics = {
                split_key: self._evaluator(model_object, *training_data[split_key]) for split_key in training_data
            }
            return model_object, hyperparameters, metrics

        self._train_task = train_task
        return train_task

    def predict_task(self):
        """Create a Flyte task for generating predictions from a model object.

        This is used in the Flyte workflow produced by ``predict_workflow``.
        """
        if self._predict_task:
            return self._predict_task

        predictor_sig = signature(self._predictor)
        model_param, *_ = predictor_sig.parameters.values()
        model_param = model_param.replace(name="model_object")

        # assume that dataset_datatype is a dict with only a single entry
        [(data_arg_name, data_arg_type)] = self._dataset.dataset_datatype.items()
        data_param = Parameter(data_arg_name, kind=Parameter.KEYWORD_ONLY, annotation=data_arg_type)

        # TODO: make sure return type is not None
        @inner_task(
            unionml_obj=self,
            input_parameters=OrderedDict([(p.name, p) for p in [model_param, data_param]]),
            return_annotation=predictor_sig.return_annotation,
            **self._predict_task_kwargs,
        )
        def predict_task(model_object, **kwargs):
            parsed_data = self.dataset._parser(kwargs[data_arg_name], **self._dataset.parser_kwargs)
            features = self.dataset._feature_transformer(parsed_data[self._dataset._parser_feature_key])
            return self._predictor(model_object, features)

        self._predict_task = predict_task
        return predict_task

    def predict_from_features_task(self):
        """Create a Flyte task for generating predictions from a model object.

        This is used in the Flyte workflow produced by ``predict_from_features_workflow``.
        """
        if self._predict_from_features_task:
            return self._predict_from_features_task

        predictor_sig = signature(self._predictor)
        model_param, *_ = predictor_sig.parameters.values()
        model_param = model_param.replace(name="model_object")

        # assume that dataset_datatype is a dict with only a single entry
        [(_, data_arg_type)] = self._dataset.dataset_datatype.items()
        data_param = Parameter("features", kind=Parameter.KEYWORD_ONLY, annotation=data_arg_type)

        @inner_task(
            unionml_obj=self,
            input_parameters=OrderedDict([("model_object", model_param), ("features", data_param)]),
            return_annotation=predictor_sig.return_annotation,
            **self._predict_task_kwargs,
        )
        def predict_from_features_task(model_object, features):
            return self._predictor(model_object, features)

        self._predict_from_features_task = predict_from_features_task
        return predict_from_features_task

    def train(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        loader_kwargs: Optional[Dict[str, Any]] = None,
        splitter_kwargs: Optional[Dict[str, Any]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **reader_kwargs,
    ) -> Tuple[Any, Any]:
        """Train a model object locally

        :param hyperparameters: a dictionary mapping hyperparameter names to values. This is passed into the
            ``init`` callable to initialize a model object.
        :param loader_kwargs: key-word arguments to pass to the registered :meth:`unionml.Dataset.loader` function.
            This will override any defaults set in the function definition.
        :param splitter_kwargs: key-word arguments to pass to the registered :meth:`unionml.Dataset.splitter` function.
            This will override any defaults set in the function definition.
        :param parser_kwargs: key-word arguments to pass to the registered :meth:`unionml.Dataset.parser` function.
            This will override any defaults set in the function definition.
        :param trainer_kwargs: a dictionary mapping training parameter names to values. There training parameters
            are determined by the keyword-only arguments of the ``model.trainer`` function.
        :param reader_kwargs: keyword arguments that correspond to the :meth:`unionml.Dataset.reader` method signature.

        The train method invokes an execution graph that composes together the following functions to train and evaluate a model:

            - :meth:`unionml.Dataset.reader`
            - :meth:`unionml.Dataset.loader`
            - :meth:`unionml.Dataset.splitter`
            - :meth:`unionml.Dataset.parser`
            - :meth:`unionml.Model.trainer`
            - :meth:`unionml.Model.predictor`
            - :meth:`unionml.Model.evaluator`

        """
        trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs
        model_obj, hyperparameters, metrics = self.train_workflow()(
            hyperparameters=self.hyperparameter_type(**({} if hyperparameters is None else hyperparameters)),
            loader_kwargs=self._dataset.loader_kwargs_type(**({} if loader_kwargs is None else loader_kwargs)),
            splitter_kwargs=self._dataset.splitter_kwargs_type(**({} if splitter_kwargs is None else splitter_kwargs)),
            parser_kwargs=self._dataset.parser_kwargs_type(**({} if parser_kwargs is None else parser_kwargs)),
            **{**reader_kwargs, **trainer_kwargs},
        )
        self.artifact = ModelArtifact(model_obj, hyperparameters, metrics)
        return model_obj, metrics

    def predict(
        self,
        features: Any = None,
        **reader_kwargs,
    ):
        """Generate predictions locally.

        You can either pass this function raw features via the ``features`` argument or you can pass in keyword
        arguments that will be forwarded to the :meth:`unionml.Dataset.reader` method as the feature source.

        :param features: Raw features that are pre-processed by the :py:class:``unionml.Dataset`` methods in the
            following order:

            - :meth:`unionml.dataset.Dataset.feature_loader`
            - :meth:`unionml.dataset.Dataset.feature_transformer`
        :param reader_kwargs: keyword arguments that correspond to the :meth:`unionml.Dataset.reader` method signature.
        """
        if features is None and not reader_kwargs:
            raise ValueError("At least one of features or **reader_kwargs needs to be provided")
        if self.artifact is None:
            raise RuntimeError(
                "ModelArtifact not found. You must train a model first with the `train` method before generating "
                "predictions."
            )
        if features is None:
            return self.predict_workflow()(model_object=self.artifact.model_object, **reader_kwargs)
        return self.predict_from_features_workflow()(
            model_object=self.artifact.model_object,
            features=self._dataset.get_features(features),
        )

    def save(self, file: Union[str, os.PathLike, IO], *args, **kwargs):
        """Save the model object to disk."""
        if self.artifact is None:
            raise AttributeError("`artifact` property is None. Call the `train` method to train a model first")
        return self._saver(self.artifact.model_object, self.artifact.hyperparameters, file, *args, **kwargs)

    def load(self, file: Union[str, os.PathLike, IO], *args, **kwargs):
        """Load a model object from disk.

        :file: a string or path-like object to load a model from.
        :param args: positional arguments forwarded to :meth:`unionml.Model.loader` .
        :param kwargs: key-word arguments forwarded to :meth:`unionml.Model.loader` .
        """
        self.artifact = ModelArtifact(self._loader(file, *args, **kwargs))
        return self.artifact.model_object

    def load_from_env(self, env_var: str = "UNIONML_MODEL_PATH", *args, **kwargs):
        """Load a model object from an environment variable pointing to the model file.

        :env_var: environment variable referencing a path to load a model from.
        :param args: positional arguments forwarded to :meth:`unionml.Model.loader` .
        :param kwargs: key-word arguments forwarded to :meth:`unionml.Model.loader` .
        """
        model_path = os.getenv(env_var)
        if model_path is None:
            raise ValueError("env_var for model path {env_var} doesn't exist.")
        return self.load(model_path)
        self.artifact = ModelArtifact(self.load(model_path))
        return self.artifact.model_object

    def serve(
        self,
        app: FastAPI,
        remote: bool = False,
        app_version: Optional[str] = None,
        model_version: str = "latest",
    ):
        """Create a FastAPI serving app.

        :param app: A ``FastAPI`` app to use for model serving.
        """
        from unionml.fastapi import serving_app

        serving_app(self, app, remote=remote, app_version=app_version, model_version=model_version)

    def remote(
        self,
        registry: Optional[str] = None,
        image_name: str = None,
        dockerfile: str = "Dockerfile",
        patch_destination_dir: str = "/root",
        config_file: Optional[str] = None,
        project: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """Configure the ``unionml.Model`` for remote backend deployment.

        :param registry: Docker registry used to push UnionML app.
        :param image_name: image name to give to the Docker image associated with the UnionML app.
        :param dockerfile: path to the Dockerfile used to package the UnionML app.
        :param patch_destination_dir: path where the UnionML source is installed within the docker image, and in case of
                                    patch registration, this would be replaced. In Flyte terms, patch registration is
                                    often called fast-registration.
        :param config_file: path to the `flytectl config <https://docs.flyte.org/projects/flytectl/en/latest/>`__ to use for
            deploying your UnionML app to a Flyte backend.
        :param project: deploy your app to this Flyte project name.
        :param domain: deploy your app to this Flyte domain name.
        """
        self._config_file = config_file
        self._registry = registry
        self._image_name = image_name
        self._dockerfile = dockerfile
        self._project = project
        self._domain = domain
        self._patch_destination_dir = patch_destination_dir

    @property
    def _remote(self) -> Optional[FlyteRemote]:
        if self.__remote__ is not None:
            return self.__remote__

        config = Config.auto(config_file=self._config_file)
        if config.platform.endpoint.startswith("localhost"):
            config = Config.for_sandbox()

        self.__remote__ = FlyteRemote(
            config=config,
            default_project=self._project,
            default_domain=self._domain,
        )
        return self.__remote__

    def remote_deploy(self, app_version: str = None, allow_uncommitted: bool = False, patch: bool = False) -> str:
        """
        Deploy model services to a Flyte backend.

        :param app_version: the version to use to deploy the UnionML app.
        :param allow_uncommitted: If True, deploys uncommitted changes in the unionml project. Otherwise, raise a
            :py:class`~unionml.remote.VersionFetchError`
        :param patch: if True, this bypasses the Docker build process and only updates the UnionML app source code using
            the latest available image.
        :returns: app version string
        """
        from unionml import remote

        if self._remote is None:
            raise RuntimeError("First configure the remote client with the `Model.remote` method")

        _remote = self._remote
        if not _remote.config.platform.endpoint.startswith("localhost") and self.registry is None:
            raise ValueError("You must specify a registry in `model.remote` when deploying to a remote cluster.")

        # if user wants to deploy by patching, then we implicitly want to allow for uncommitted changes.
        is_explicit_app_version = app_version is not None
        app_version = app_version or remote.get_app_version(allow_uncommitted=allow_uncommitted or patch)
        image = remote.get_image_fqn(self, app_version, self._image_name)
        os.environ["FLYTE_INTERNAL_IMAGE"] = image or ""

        logger.info(f"Using remote endpoint {_remote.config.platform.endpoint}")
        remote.create_project(_remote, self._project)
        if patch:
            app_version = app_version if is_explicit_app_version else f"{app_version}-patch{uuid.uuid4().hex[:7]}"
        else:
            logger.info(
                "Building docker image. If no dependencies have changed, use the `patch` arg to bypass "
                "docker build process"
            )
            if _remote.config.platform.endpoint.startswith("localhost"):
                # assume that a localhost flyte_admin_url means that we want to use Flyte sandbox
                remote.sandbox_docker_build(self, image)
            else:
                remote.docker_build_push(self, image)

        logger.info(f"Deploying app version {app_version}")
        for wf in [
            self.train_workflow(),
            self.predict_workflow(),
            self.predict_from_features_workflow(),
        ]:
            remote.deploy_wf(
                wf,
                _remote,
                image,
                project=_remote.default_project,
                domain=_remote.default_domain,
                version=app_version,
                patch=patch,
                patch_destination_dir=self._patch_destination_dir,
            )

        return app_version

    def remote_train(
        self,
        app_version: str = None,
        wait: bool = True,
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
        loader_kwargs: Optional[Dict[str, Any]] = None,
        splitter_kwargs: Optional[Dict[str, Any]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **reader_kwargs,
    ) -> Union[ModelArtifact, FlyteWorkflowExecution]:
        """Train a model object on a remote Flyte backend.

        :param app_version: if provided, executes a training job using the specified UnionML app version. By default,
            this uses the latest app version deployed on the Flyte remote cluster..
        :param wait: if True, this is a synchronous operation, returning a ``ModelArtifact``. Otherwise, this
            function returns a ``FlyteWorkflowExecution``.
        :param hyperparameters: a dictionary mapping hyperparameter names to values. This is passed into the
            ``init`` callable to initialize a model object.
        :param loader_kwargs: key-word arguments to pass to the registered :meth:`unionml.Dataset.loader` function.
            This will override any defaults set in the function definition.
        :param splitter_kwargs: key-word arguments to pass to the registered :meth:`unionml.Dataset.splitter` function.
            This will override any defaults set in the function definition.
        :param parser_kwargs: key-word arguments to pass to the registered :meth:`unionml.Dataset.parser` function.
            This will override any defaults set in the function definition.
        :param trainer_kwargs: a dictionary mapping training parameter names to values. There training parameters
            are determined by the keyword-only arguments of the ``model.trainer`` function.
        :param reader_kwargs: keyword arguments that correspond to the :meth:`unionml.Dataset.reader` method signature.
        :returns: the trained model if wait is ``True``, or a ``FlyteWorkflowExecution`` object if wait is ``False``.
        """
        if self._remote is None:
            raise RuntimeError("First configure the remote client with the `Model.remote` method")

        train_wf = self._remote.fetch_workflow(name=self.train_workflow_name, version=app_version)
        execution = self._remote.execute(
            train_wf,
            inputs={
                "hyperparameters": self.hyperparameter_type(**({} if hyperparameters is None else hyperparameters)),
                "loader_kwargs": self._dataset.loader_kwargs_type(**loader_kwargs or {}),
                "splitter_kwargs": self._dataset.splitter_kwargs_type(**splitter_kwargs or {}),
                "parser_kwargs": self._dataset.parser_kwargs_type(**parser_kwargs or {}),
                **{**reader_kwargs, **({} if trainer_kwargs is None else trainer_kwargs)},  # type: ignore
            },
            project=self._remote.default_project,
            domain=self._remote.default_domain,
            type_hints={
                "hyperparameters": self.hyperparameter_type,
                "loader_kwargs": self._dataset.loader_kwargs_type,
                "splitter_kwargs": self._dataset.splitter_kwargs_type,
                "parser_kwargs": self._dataset.parser_kwargs_type,
            },
        )
        console_url = self._remote.generate_console_url(execution)
        logger.info(
            f"Executing {train_wf.id.name}, execution name: {execution.id.name}."
            f"\nGo to {console_url} to see the execution in the console."
        )

        if not wait:
            return execution

        self.remote_wait(execution)
        self.remote_load(execution)
        return self.artifact

    def remote_predict(
        self,
        app_version: str = None,
        model_version: str = None,
        wait: bool = True,
        *,
        features: Any = None,
        **reader_kwargs,
    ) -> Union[Any, FlyteWorkflowExecution]:
        """Generate predictions on a remote Flyte backend.

        You can either pass this function raw features via the ``features`` argument or you can pass in keyword
        arguments that will be forwarded to the :meth:`unionml.Dataset.reader` method as the feature source.

        :param app_version: if provided, executes a prediction job using the specified UnionML app version. By default,
            this uses the latest app version deployed on the Flyte remote cluster.
        :param model_version: if provided, executes a prediction job using the specified model version. By default, this
            uses the latest Flyte execution id as the model version.
        :param wait: if True, this is a synchronous operation, returning a ``ModelArtifact``. Otherwise, this
            function returns a ``FlyteWorkflowExecution``.
        :param features: Raw features that are pre-processed by the :py:class:``unionml.Dataset`` methods in the
            following order:

            - :meth:`unionml.dataset.Dataset.feature_loader`
            - :meth:`unionml.dataset.Dataset.feature_transformer`
        :param reader_kwargs: keyword arguments that correspond to the :meth:`unionml.Dataset.reader` method signature.
        :returns: the predictions if wait is ``True``, or a ``FlyteWorkflowExecution`` object if wait is ``False``.
        """
        if self._remote is None:
            raise RuntimeError("First configure the remote client with the `Model.remote` method")

        from unionml import remote

        model_artifact = remote.get_model_artifact(self, app_version, model_version)
        inputs = {"model_object": model_artifact.model_object}
        if features is None:
            workflow_name = self.predict_workflow_name
            inputs.update(reader_kwargs)
            type_hints = {}
        else:
            workflow_name = self.predict_from_features_workflow_name
            inputs.update({"features": self._dataset.get_features(features)})
            type_hints = {"features": [*self._dataset.dataset_datatype.values()][0]}

        predict_wf = self._remote.fetch_workflow(
            self._remote._default_project,
            self._remote._default_domain,
            workflow_name,
            app_version,
        )
        execution = self._remote.execute(
            predict_wf,
            inputs=inputs,
            project=self._remote.default_project,
            domain=self._remote.default_domain,
            wait=wait,
            type_hints=type_hints,
        )
        console_url = self._remote.generate_console_url(execution)
        logger.info(
            f"Executing {predict_wf.id.name}, execution name: {execution.id.name}."
            f"\nGo to {console_url} to see the execution in the console."
        )
        if not wait:
            return execution
        predictions, *_ = execution.outputs.values()
        return predictions

    def remote_wait(self, execution: FlyteWorkflowExecution, **kwargs) -> Any:
        """Wait for a ``FlyteWorkflowExecution`` to complete and returns the execution's output."""
        if self._remote is None:
            raise ValueError("You must call `model.remote` to attach a remote backend to this model.")
        return self._remote.wait(execution, **kwargs)

    def remote_load(self, execution: FlyteWorkflowExecution):
        """Load a ``ModelArtifact`` based on the provided Flyte execution.

        :param execution: a Flyte workflow execution, which is the output of ``remote_train(..., wait=False)`` .
        """
        if self._remote is None:
            raise ValueError("You must call `model.remote` to attach a remote backend to this model.")
        if not execution.is_done:
            logger.info(f"Waiting for execution {execution.id.name} to complete...")
            execution = self.remote_wait(execution)
            logger.info("Done.")

        with self._remote.remote_context():
            try:
                model_object = execution.outputs.get("model_object", as_type=self.model_type)
            except ValueError:
                model_object = execution.outputs.get("model_object", as_type=FlytePickle)

            self.artifact = ModelArtifact(
                model_object,
                execution.outputs["hyperparameters"],
                execution.outputs["metrics"],
            )

    def remote_list_model_versions(self, app_version: str = None, limit: int = 10) -> List[str]:
        """Lists all the model versions of this UnionML app, in reverse chronological order.

        :param app_version: if provided, lists the model versions associated with this app version. By default,
            this uses the current git sha of the repo, which versions your UnionML app.
        :param limit: limit the number results to fetch.
        """
        from unionml import remote

        app_version = app_version or remote.get_app_version(allow_uncommitted=True)
        return remote.list_model_versions(self, app_version=app_version, limit=limit)

    def remote_fetch_predictions(self, execution: FlyteWorkflowExecution) -> Any:
        """Fetch predictions from a Flyte execution.

        :param execution: a Flyte workflow execution, which is the output of ``remote_predict(..., wait=False)`` .
        """
        if self._remote is None:
            raise ValueError("You must call `model.remote` to attach a remote backend to this model.")
        execution = self._remote.wait(execution)
        predictions, *_ = execution.outputs.values()
        return predictions

    @property
    def model_type(self) -> Type:
        init = self._init_callable if self._init == self._default_init else self._init or self._init_callable
        return init if inspect.isclass(init) else signature(init).return_annotation if init is not None else init

    def _default_init(self, hyperparameters: dict) -> Any:
        if self._init_callable is None:
            raise ValueError(
                "When using the _default_init method, you must specify the init argument to the Model constructor."
            )
        return self._init_callable(**hyperparameters)

    def _default_saver(
        self,
        model_obj: Any,
        hyperparameters: Union[dict, BaseHyperparameters, None],
        file: Union[str, os.PathLike, IO],
        *args,
        **kwargs,
    ) -> Any:
        model_type = self.model_type
        hyperparameters = (
            asdict(hyperparameters)
            if hyperparameters is not None and is_dataclass(hyperparameters)
            else hyperparameters
        )
        if isinstance(model_obj, sklearn.base.BaseEstimator):
            return joblib.dump({"model_obj": model_obj, "hyperparameters": hyperparameters}, file, *args, **kwargs)
        elif is_pytorch_model(model_type):
            import torch

            torch.save(
                {"model_obj": model_obj.state_dict(), "hyperparameters": hyperparameters},
                file,
                *args,
                **kwargs,
            )
            return file
        elif is_keras_model(model_type):
            model_obj.save(file, *args, **kwargs)
            return file

        raise NotImplementedError(
            f"Default saver not defined for type {type(model_obj)}. Use the Model.saver decorator to define one."
        )

    def _default_loader(self, file: Union[str, os.PathLike, IO], *args, **kwargs) -> Any:
        model_type = self.model_type
        if issubclass(model_type, sklearn.base.BaseEstimator):
            deserialized_model = joblib.load(file, *args, **kwargs)
            return deserialized_model["model_obj"]
        elif is_pytorch_model(model_type):
            import torch

            deserialized_model = torch.load(file, *args, **kwargs)
            if self._init_callable is not None:
                model = self._init(deserialized_model["hyperparameters"])
            else:
                model = model_type(**deserialized_model["hyperparameters"])

            model.load_state_dict(deserialized_model["model_obj"])
            return model
        elif is_keras_model(model_type):
            from tensorflow import keras

            return keras.models.load_model(file)

        raise NotImplementedError(
            f"Default loader not defined for type {model_type}. Use the Model.loader decorator to define one."
        )
