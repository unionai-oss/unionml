import os

from fastapi import FastAPI

from tests.integration.sklearn.quickstart import model

app = FastAPI()
model.serve(app, model_path=os.getenv("TEST_FKLEARN_MODEL_PATH"))
