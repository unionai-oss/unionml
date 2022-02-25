from fastapi import FastAPI

from tests.integration.sklearn.quickstart import model

app = FastAPI()
model.serve(app)
