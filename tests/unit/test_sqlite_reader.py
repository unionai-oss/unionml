import pytest
import sqlite3
import typing

import pandas as pd
from flytekit import kwtypes
from flytekit.extras.sqlite3.task import SQLite3Config, SQLite3Task
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from flytekit_learn import Dataset, Model


@pytest.fixture
def mock_sqlite_db(tmp_path):
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0] * 10,
        "x2": [1.0, 2.0, 3.0] * 10,
        "x3": [1.0, 2.0, 3.0] * 10,
        "y": [1, 0, 1] * 10,
    })
    file = tmp_path / "test_dataset.sqlite"
    with sqlite3.connect(file) as conn:
        df.to_sql("mock_table", conn, index=False)
        yield file


def test_sqlite_dataset_reader(mock_sqlite_db):

    sqlite_task = SQLite3Task(
        name="test_sqlite_db",
        query_template="""
        SELECT x1, x2, x3, y FROM mock_table LIMIT {{.inputs.limit}}
        """,
        inputs=kwtypes(limit=int),
        output_schema_type=pd.DataFrame,
        task_config=SQLite3Config(uri=mock_sqlite_db.as_uri())
    )

    limit = 50

    with sqlite3.connect(mock_sqlite_db) as conn:
        expected_df = pd.read_sql(f"SELECT x1, x2, x3, y FROM mock_table LIMIT {limit}", conn)

    task_output_df = sqlite_task(limit=limit)
    assert task_output_df.equals(expected_df)

    dataset = Dataset.from_sqlite_task(
        sqlite_task,
        name="test_dataset",
        features=["x1", "x2", "x3"],
        targets=["y"],
    )
    model = Model(
        name="test_model",
        init=LogisticRegression,
        dataset=dataset,
    )

    @model.trainer
    def trainer(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
        return model.fit(features, target.squeeze())
    
    @model.predictor
    def predictor(model: LogisticRegression, features: pd.DataFrame) -> typing.List[float]:
        return [float(x) for x in model.predict(features)]

    @model.evaluator
    def evaluator(model: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> float:
        predictions = model.predict(features)
        return accuracy_score(target, predictions)

    trained_model, _ = model.train(hyperparameters={"C": 100.0}, limit=50)
    assert isinstance(trained_model, LogisticRegression)

    predictions = model.predict(trained_model, limit=5)
    features = pd.read_sql(f"SELECT x1, x2, x3 FROM mock_table LIMIT 5", conn)
    alt_predictions = model.predict(trained_model, features=features)
    assert all(isinstance(x, float) for x in predictions)
    assert predictions == alt_predictions
