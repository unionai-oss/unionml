import requests
from sklearn.datasets import load_digits

digits = load_digits(as_frame=True)
features = digits.frame[digits.feature_names]
n_samples = 5

prediction_response = requests.post(
    "http://0.0.0.0:3033/predict",
    json=features.sample(n_samples, random_state=42).to_dict(orient="records"),
)

print(prediction_response.text)
