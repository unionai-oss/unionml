"""Utilities for the FastAPI integration."""

import os
from typing import Any, Dict, List, Optional, Union

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from flytekit_learn.model import Model, ModelArtifact
from flytekit_learn.remote import get_latest_model_artifact


class TrainParams:

    __slots__ = ("model", "remote", "local", "wait", "inputs")

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        local: bool = False,
        wait: bool = False,
        inputs: Optional[Union[Dict, BaseModel]] = Body(None),
    ) -> "TrainParams":
        self.remote = self.model._remote
        self.local = local
        self.wait = wait
        self.inputs = inputs
        return self


class PredictParams:

    __slots__ = (
        "model",
        "remote",
        "local",
        "model_version",
        "inputs",
        "features",
    )

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        local: bool = False,  # TODO remove this, instead do model.serve(app, debug=True)
        model_version: str = "latest",
        inputs: Optional[Union[Dict, BaseModel]] = Body(None),
        features: Optional[List[Dict[str, Any]]] = Body(None),
    ) -> "PredictParams":
        self.remote = self.model._remote
        self.local = local
        self.model_version = model_version
        self.inputs = inputs
        self.features = features
        return self


def serving_app(model: Model, app: FastAPI):
    # TODO: load a model from a flytebackend here
    model_path = os.getenv("FKLEARN_MODEL_PATH")
    if model_path:
        model.artifact = ModelArtifact(model.load(model_path))

    @app.get("/", response_class=HTMLResponse)
    def root():
        return """
            <html>
                <head>
                    <title>flytekit-learn</title>
                </head>
                <body>
                    <h1>flytekit-learn</h1>
                    <p>The easiest way to build and deploy models</p>
                </body>
            </html>
        """

    @app.post("/predict")
    def predict(params: PredictParams = Depends(PredictParams(model))):
        inputs, features = params.inputs, params.features
        if inputs is None and features is None:
            raise HTTPException(status_code=500, detail="inputs or features must be supplied.")

        version = None if params.model_version == "latest" else params.model_version
        if not params.local:
            model.artifact = get_latest_model_artifact(model, version)

        if model.artifact is None:
            raise HTTPException(status_code=500, detail="trained model not found")

        workflow_inputs: Dict[str, Any] = {}
        workflow_inputs.update(inputs if inputs else {"features": model._dataset.get_features(features)})
        if params.local:
            return model.predict(**workflow_inputs)

        predict_wf = params.remote.fetch_workflow(
            name=model.predict_workflow_name if inputs else model.predict_from_features_workflow_name,
            version=version,
        )
        return params.remote.execute(predict_wf, inputs=workflow_inputs, wait=True).outputs["o0"]
