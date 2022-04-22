from fastapi import FastAPI

from tests.integration.sklearn_app.quickstart import model

app = FastAPI()
model.serve(app)
