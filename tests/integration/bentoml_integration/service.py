from tests.integration.sklearn_app.quickstart import model
from unionml.services.bentoml import BentoMLService

bentoml_service = BentoMLService(model)

model_object, _ = model.train(hyperparameters={"C": 1.0, "max_iter": 10000})
bentoml_service.serve(model_object=model_object)
