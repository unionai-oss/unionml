import requests
from sklearn.datasets import load_digits

digits = load_digits(as_frame=True)
features = digits.frame[digits.feature_names]


prediction_response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": features.sample(5, random_state=42).to_dict(orient="records")},
)

print(prediction_response.text)
