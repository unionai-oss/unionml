import requests
from sklearn.datasets import load_digits

df = load_digits(as_frame=True).frame.drop(["target"], axis="columns")

r = requests.post(
    "http://0.0.0.0:3000/predict",
    headers={"content-type": "application/json"},
    data=df.sample(5, random_state=42).to_json(orient="records"),
)
print(r.text)
