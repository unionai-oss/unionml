"""Utilities for the FastAPI integration."""

import os
from typing import Any, Dict, List, NamedTuple, Optional, Union

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from flytekit.remote import FlyteRemote
from pydantic import BaseModel

from unionml.model import Model, ModelArtifact
from unionml.remote import get_latest_model_artifact


class PredictParams(NamedTuple):
    model: Model
    remote: FlyteRemote
    local: bool
    model_version: str
    inputs: Optional[Union[Dict, BaseModel]]
    features: Optional[List[Dict[str, Any]]]


def serving_app(model: Model, app: FastAPI):
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

    async def load_predict_params(
        local: bool = False,
        model_version: str = "latest",
        inputs: Optional[Union[Dict, BaseModel]] = Body(None),
        features: Optional[List[Dict[str, Any]]] = Body(None),
    ) -> PredictParams:
        # TODO: load a model from a flytebackend here
        model_path = os.getenv("unionml_MODEL_PATH")
        if model_path:
            model.artifact = ModelArtifact(model.load(model_path))
        return PredictParams(model, model._remote, local, model_version, inputs, features)

    @app.post("/predict")
    async def predict(params: PredictParams = Depends(load_predict_params)):
        inputs, features = params.inputs, params.features
        if inputs is None and features is None:
            raise HTTPException(status_code=500, detail="inputs or features must be supplied.")

        version = None if params.model_version == "latest" else params.model_version
        if not params.local:
            model.artifact = get_latest_model_artifact(model, version)

        if model.artifact is None:
            raise HTTPException(status_code=500, detail="trained model not found")

        workflow_inputs: Dict[str, Any] = {}
        if model._dataset.reader_return_type is not None:
            # convert raw features to whatever the output type of the reader is.
            (_, feature_type), *_ = model._dataset.reader_return_type.items()
            features = feature_type(features)
        workflow_inputs.update(inputs if inputs else {"features": features})
        if params.local:
            return model.predict(**workflow_inputs)

        predict_wf = params.remote.fetch_workflow(
            name=model.predict_workflow_name if inputs else model.predict_from_features_workflow_name,
            version=version,
        )
        return params.remote.execute(predict_wf, inputs=workflow_inputs, wait=True).outputs["o0"]
