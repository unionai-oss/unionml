import requests
from sklearn.datasets import load_digits

digits = load_digits(as_frame=True)
features = digits.frame[digits.feature_names]

requests.post(
    "http://127.0.0.1:8000/train?local=True",
    json={"hyperparameters": {"C": 1.0, "max_iter": 1000}},
)

requests.get(
    "http://127.0.0.1:8000/predict?local=True",
    json={"features": features.sample(5, random_state=42).to_dict(orient="records")},
)
