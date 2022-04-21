from sklearn.datasets import load_breast_cancer

from example.app.main import model

model.remote_deploy()
model_artifact = model.remote_train(hyperparameters={"C": 1.0}, random_state=42, sample_frac=1.0)
predictions = model.remote_predict(random_state=42, sample_frac=0.05)

features = load_breast_cancer(as_frame=True).frame.sample(5, random_state=42)
predictions_from_features = model.remote_predict(features=features)
