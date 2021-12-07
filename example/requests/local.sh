# !/bin/sh

curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"hyperparameters": {"C": 1.0, "max_iter": 1000}}' \
    "http://127.0.0.1:8000/train?local=True"

curl -X GET \
    -H "Content-Type: application/json" \
    -d "{\"features\": $(cat ./example/data/sample_breast_cancer_data.json)}" \
    "http://127.0.0.1:8000/predict?local=True&model_source=local"
