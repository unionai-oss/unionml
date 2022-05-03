"""Utilities for the FastAPI integration."""

import os
from typing import Any, Dict, List, NamedTuple, Optional, Union

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from unionml.model import Model, ModelArtifact
from unionml.remote import get_model_artifact


class PredictParams(NamedTuple):
    model_version: str
    inputs: Optional[Union[Dict, BaseModel]]
    features: Optional[List[Dict[str, Any]]]


class ModelArtifactCache:
    def __init__(self, model: Model, max_size: int = 100):
        self.model = model
        self.max_size = max_size
        self.cache_from_path: Dict[str, ModelArtifact] = {}
        self.cache_from_version: Dict[str, ModelArtifact] = {}

    def get_from_path(self, model_path: str):
        artifact = self.cache_from_path.get(model_path)
        if artifact is not None:
            return artifact
        artifact = ModelArtifact(self.model.load(model_path))
        self.cache_from_path[model_path] = artifact
        self.truncate(self.cache_from_path)
        return artifact

    def get_from_version(self, model_version: str):
        if model_version == "latest":
            return get_model_artifact(self.model, model_version=None)

        artifact = self.cache_from_version.get(model_version)
        if artifact is not None:
            return artifact
        artifact = get_model_artifact(self.model, model_version=model_version)
        self.cache_from_version[model_version] = artifact
        self.truncate(self.cache_from_version)
        return artifact

    def truncate(self, cache: dict):
        if len(self.cache_from_version) > self.max_size:
            self.truncate(self.cache_from_version)
            # since dicts preserve insertion order
            delete_key = list(cache.keys())[0]
            cache.pop(delete_key)


def serving_app(model: Model, app: FastAPI, remote: bool = False):
    @app.get("/", response_class=HTMLResponse)
    def root():
        return """
            <html>
                <head>
                    <title>unionml</title>
                </head>
                <body>
                    <h1>unionml</h1>
                    <p>The easiest way to build and deploy models</p>
                </body>
            </html>
        """

    model_artifact_cache = ModelArtifactCache(model)

    async def load_predict_params(
        model_version: str = "latest",
        inputs: Optional[Union[Dict, BaseModel]] = Body(None),
        features: Optional[List[Dict[str, Any]]] = Body(None),
    ) -> PredictParams:
        model_path = os.getenv("UNIONML_MODEL_PATH")
        if model_path or not remote:
            if model_path is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Model artifact path not specified. Make sure to specify the unionml serve --model-path in "
                        "the option when starting the unionml prediction service in local mode."
                    ),
                )
            model.artifact = model_artifact_cache.get_from_path(model_path)
        else:
            model.artifact = model_artifact_cache.get_from_version(model_version)
        return PredictParams(model_version, inputs, features)

    @app.post("/predict")
    async def predict(params: PredictParams = Depends(load_predict_params)):
        inputs, features = params.inputs, params.features
        if inputs is None and features is None:
            raise HTTPException(status_code=500, detail="inputs or features must be supplied.")

        workflow_inputs: Dict[str, Any] = {}
        if model._dataset.reader_return_type is not None:
            # convert raw features to whatever the output type of the reader is.
            (_, feature_type), *_ = model._dataset.reader_return_type.items()
            features = feature_type(features)
        workflow_inputs.update(inputs if inputs else {"features": features})

        return model.predict(**workflow_inputs)
