from fastapi import FastAPI

from tests.integration.pytorch_app.quickstart import model

app = FastAPI()
model.serve(app)
