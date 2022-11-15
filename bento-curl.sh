#!/bin/sh
curl -X POST \
    -H "Content-Type: application/json" \
    --data "$(cat sample_features.json)" \
    http://0.0.0.0:3000/predict
