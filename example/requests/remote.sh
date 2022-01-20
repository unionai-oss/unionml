#!/bin/sh

curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"hyperparameters": {"C": 1.0, "max_iter": 1000}, "sample_frac": 1.0, "random_state": 123}' \
    "http://127.0.0.1:8000/train"

curl -X GET \
    -H "Content-Type: application/json" \
    -d '{"inputs": {"sample_frac": 0.01, "random_state": 43}}' \
    "http://127.0.0.1:8000/predict"

curl -X GET \
    -H "Content-Type: application/json" \
    -d "{\"features\": $(cat ./example/data/sample_breast_cancer_data.json)}" \
    "http://127.0.0.1:8000/predict"
