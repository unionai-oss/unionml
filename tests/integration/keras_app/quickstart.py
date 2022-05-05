from typing import List

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import Sequential

from unionml import Dataset, Model


# define a simple keras model
def build_keras_model(
    in_dims: int,
    hidden_dims: int,
    out_dims: int,
) -> Sequential:
    keras_model = Sequential()
    keras_model.add(keras.layers.Dense(hidden_dims, input_shape=(in_dims,), activation="relu"))
    keras_model.add(keras.layers.Dense(out_dims, activation="softmax"))
    return keras_model


dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=build_keras_model, dataset=dataset)


@dataset.reader
def reader() -> pd.DataFrame:
    return load_digits(as_frame=True).frame


@model.trainer
def trainer(
    keras_model: Sequential,
    features: pd.DataFrame,
    target: pd.DataFrame,
    *,
    # keyword-only arguments define trainer parameters
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
) -> Sequential:
    keras_model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
        run_eagerly=True,
    )
    keras_model.fit(
        features.values,
        keras.utils.to_categorical(target.values),
        batch_size=batch_size,
        epochs=n_epochs,
    )
    return keras_model


@model.predictor
def predictor(keras_model: Sequential, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in keras_model.predict(features).argmax(1)]


@model.evaluator
def evaluator(keras_model: Sequential, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return float(accuracy_score(target.squeeze(), predictor(keras_model, features)))


if __name__ == "__main__":
    model_object, metrics = model.train(
        hyperparameters={"in_dims": 64, "hidden_dims": 32, "out_dims": 10},
        trainer_kwargs={"batch_size": 512, "n_epochs": 100, "learning_rate": 0.0003},
    )
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # save model to a file using torch.save
    model.save("/tmp/model_object.h5")
