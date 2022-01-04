"""Utilities for the FastAPI integration."""

from enum import Enum
from functools import wraps
from inspect import signature
from typing import Any, Dict, List, Optional

from fastapi import Body, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, create_model

from flytekit.models import filters
from flytekit.models.admin.common import Sort
from flytekit.remote import FlyteWorkflowExecution


class TrainParams:

    __slots__ = ("model", "remote", "local", "wait", "inputs")

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        local: bool = False,
        wait: bool = False,
        inputs: Optional[Dict] = Body(None),
    ) -> "TrainParams":
        self.remote = self.model._remote
        self.local = local
        self.wait = wait
        self.inputs = inputs
        return self


class PredictParams:

    __slots__ = ("model", "remote", "local", "model_version", "model_source", "inputs", "features")

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        local: bool = False,  # TODO remove this, instead do model.serve(app, debug=True)
        model_version: str = "latest",
        model_source: str = "remote",  # TODO remove this, model source should be flyte backend if debug=False
        inputs: Optional[Dict] = Body(None),
        features: Optional[List[Dict[str, Any]]] = Body(None),
    ) -> "PredictParams":
        self.remote = self.model._remote
        self.local = local
        self.model_version = model_version
        self.model_source = model_source
        self.inputs = inputs
        self.features = features
        return self


def app_wrapper(model, app, default_endpoints: bool, train_endpoint: str, predict_endpoint: str):
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

    if not default_endpoints:
        return

    @app.post(train_endpoint)
    def train(params: TrainParams = Depends(TrainParams(model))):
        inputs = params.inputs.dict() if issubclass(type(params.inputs), BaseModel) else params.inputs
        if params.local:
            trained_model, metrics = model.train(**inputs)
            model._latest_model = trained_model
            model._metrics = metrics
            trained_model, flyte_execution_id = str(trained_model), None
        else:
            train_wf = params.remote.fetch_workflow(name=params.model.train_workflow_name)
            execution = params.remote.execute(train_wf, inputs=inputs, wait=params.wait)
            flyte_execution_id = {
                "project": execution.id.project,
                "domain": execution.id.domain,
                "name": execution.id.name
            }
            trained_model, metrics = None, None
            if params.wait:
                trained_model = execution.outputs["trained_model"]
                metrics = execution.outputs["metrics"]
        return {
            "trained_model": trained_model,
            "metrics": metrics,
            "flyte_execution_id": flyte_execution_id
        }

    @app.get(predict_endpoint)
    def predict(params: PredictParams = Depends(PredictParams(model))):
        inputs, features = params.inputs, params.features
        if inputs is None and features is None:
            raise HTTPException(status_code=500, detail="inputs or features must be supplied.")

        version = None if params.model_version == "latest" else params.model_version
        if params.model_source == "remote":
            latest_model = _get_latest_trained_model(model, params.remote, version)
        elif model._latest_model:
            latest_model = model._latest_model
        else:
            raise HTTPException(status_code=500, detail="trained model not found")

        workflow_inputs = {"model": latest_model}
        workflow_inputs.update(inputs if inputs else {"features": model._dataset.get_features(features)})
        if params.local:
            return model.predict(**workflow_inputs)

        predict_wf = params.remote.fetch_workflow(
            name=model.predict_workflow_name if inputs else model.predict_from_features_workflow_name,
            version=version,
        )
        return params.remote.execute(predict_wf, inputs=workflow_inputs, wait=True).outputs["o0"]


def _get_latest_trained_model(model, remote, version):
    train_workflow = remote.fetch_workflow(name=model.train_workflow_name, version=version)
    [latest_training_execution, *_], _ = remote.client.list_executions_paginated(
        train_workflow.id.project,
        train_workflow.id.domain,
        limit=1,
        filters=[
            filters.Equal("launch_plan.name", train_workflow.id.name),
            filters.Equal("phase", "SUCCEEDED"),
        ],
        sort_by=Sort("created_at", Sort.Direction.DESCENDING),
    )
    latest_training_execution = FlyteWorkflowExecution.promote_from_model(latest_training_execution)
    remote.sync(latest_training_execution)
    return latest_training_execution.outputs["trained_model"]
