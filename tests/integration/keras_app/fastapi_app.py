from fastapi import FastAPI

from tests.integration.keras_app.quickstart import model

app = FastAPI()
model.serve(app)
