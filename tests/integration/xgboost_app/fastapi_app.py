from fastapi import FastAPI

from tests.integration.xgboost_app.quickstart import model

app = FastAPI()
model.serve(app)
