import requests

import pandas as pd
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer(as_frame=True)
training_data = breast_cancer_data.frame
features = training_data[breast_cancer_data.feature_names]

while True:
    unlabelled_batch = requests.post(
        "http://127.0.0.1:8000/label/session?batch_size=1",
        json={"reader_inputs": {"sample_frac": 0.01, "random_state": 123}},
    )
    unlabelled_batch_json = unlabelled_batch.json()
    if unlabelled_batch.status_code == 400 or unlabelled_batch_json["session_complete"]:
        print(unlabelled_batch.status_code, unlabelled_batch.json())
        break
    unlabelled_batch_df = pd.DataFrame(unlabelled_batch_json["batch"])
    print(f"Unlabelled_batch: {unlabelled_batch_json['batch']}")

    # label the unlabelled data
    labelled_batch = unlabelled_batch_df.assign(target=1)

    submission_response = requests.post(
        "http://127.0.0.1:8000/label/session?submit=True",
        json={"submission": labelled_batch.to_dict(orient="records")},
    )
    print(submission_response.json())
