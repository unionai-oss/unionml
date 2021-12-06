import requests

from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer(as_frame=True)
training_data = breast_cancer_data.frame
features = training_data[breast_cancer_data.feature_names]

metrics = requests.post(
    "http://127.0.0.1:8000/train?remote=True",
    json={"hyperparameters": {"C": 1.0, "max_iter": 1000}}
)
print(f"Model: {metrics.text}")

predictions = requests.get(
    "http://127.0.0.1:8000/predict?remote=True",
    json={"features": features.sample(10, random_state=42).to_dict(orient="records")}
)
print(f"Predictions: {predictions.text}")
