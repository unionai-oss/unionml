import requests
from sklearn.datasets import load_digits

digits = load_digits(as_frame=True)
features = digits.frame[digits.feature_names]

training_response = requests.post(
    "http://127.0.0.1:8000/train?local=True",
    json={"hyperparameters": {"C": 1.0, "max_iter": 1000}},
)

prediction_response = requests.post(
    "http://127.0.0.1:8000/predict?local=True",
    json={"features": features.sample(5, random_state=42).to_dict(orient="records")},
)

print(training_response.text)
print(prediction_response.text)
