import requests
from sklearn.datasets import load_digits

digits = load_digits(as_frame=True)
features = digits.frame[digits.feature_names]


prediction_response = requests.post(
    "https://b7qct7cz79.execute-api.us-east-2.amazonaws.com/Prod/predict",
    json={"features": features.sample(5, random_state=42).to_dict(orient="records")},
)

print(prediction_response.text)
