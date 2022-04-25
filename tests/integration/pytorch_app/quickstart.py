from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from ulearn import Dataset, Model


# define a simple pytorch module
class PytorchModel(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int, out_dims: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, out_dims),
        )

    def forward(self, features):
        return F.softmax(self.layers(features), dim=1)


dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=PytorchModel, dataset=dataset)


def process_features(features: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(features.values).float()


def process_target(target: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(target.squeeze().values).long()


@dataset.reader
def reader() -> pd.DataFrame:
    return load_digits(as_frame=True).frame


@model.trainer
def trainer(
    module: PytorchModel,
    features: pd.DataFrame,
    target: pd.DataFrame,
    *,
    # keyword-only arguments define trainer parameters
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    weight_decay: float
) -> PytorchModel:
    opt = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(n_epochs):
        for (X, y) in zip(
            torch.split(process_features(features), batch_size),
            torch.split(process_target(target), batch_size),
        ):
            opt.zero_grad()
            loss = F.cross_entropy(module(X), y)
            loss.backward()
            opt.step()
    return module


@model.predictor
def predictor(module: PytorchModel, features: pd.DataFrame) -> List[float]:
    return [float(x) for x in module(process_features(features)).argmax(1)]


@model.evaluator
def evaluator(module: PytorchModel, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return float(accuracy_score(target.squeeze(), predictor(module, features)))


if __name__ == "__main__":
    trained_model, metrics = model.train(
        hyperparameters={"in_dims": 64, "hidden_dims": 32, "out_dims": 10},
        trainer_kwargs={"batch_size": 512, "n_epochs": 1000, "learning_rate": 0.0003, "weight_decay": 0.0001},
    )
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(trained_model, metrics, predictions, sep="\n")

    # save model to a file using torch.save
    model.save("/tmp/model_object.pt")
