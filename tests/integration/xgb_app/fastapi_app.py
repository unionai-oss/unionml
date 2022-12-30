from fastapi import FastAPI

from tests.integration.xgb_app.quickstart import model

app = FastAPI()
model.serve(app)
