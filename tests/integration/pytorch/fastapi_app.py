from fastapi import FastAPI

from tests.integration.pytorch.quickstart import model

app = FastAPI()
model.serve(app)
