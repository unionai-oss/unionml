from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from flytekit_learn import Dataset, Model


class PytorchModel(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int, out_dims: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, out_dims),
        )

    def forward(self, features):
        return F.softmax(self.layers(features), dim=1)


dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=PytorchModel, dataset=dataset)


@dataset.reader
def reader() -> pd.DataFrame:
    return load_digits(as_frame=True).frame


@dataset.parser
def parser(
    data: pd.DataFrame,
    features: Optional[List[str]],
    targets: List[str],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not features:
        features = [col for col in data.columns if col not in targets]
    try:
        target_data = torch.from_numpy(data[targets].squeeze().values).long()
    except KeyError:
        target_data = None
    return (torch.from_numpy(data[features].values).float(), target_data)


@model.trainer
def trainer(
    pytorch_model: PytorchModel,
    features: pd.DataFrame,
    target: pd.DataFrame,
    *,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    weight_decay: float
) -> PytorchModel:
    opt = torch.optim.Adam(pytorch_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(n_epochs):
        for (X, y) in zip(torch.split(features, batch_size), torch.split(target, batch_size)):
            opt.zero_grad()
            out = pytorch_model(X)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
    return pytorch_model


@model.predictor
def predictor(pytorch_model: PytorchModel, features: torch.Tensor) -> List[float]:
    return [float(x) for x in pytorch_model(features).argmax(1)]


@model.evaluator
def evaluator(pytorch_model: PytorchModel, features: torch.Tensor, target: torch.Tensor) -> float:
    return accuracy_score(target.squeeze(), predictor(pytorch_model, features))


if __name__ == "__main__":
    trained_model, metrics = model.train(
        hyperparameters={"in_dims": 64, "hidden_dims": 32, "out_dims": 10},
        trainer_kwargs={"batch_size": 512, "n_epochs": 1000, "learning_rate": 0.0003, "weight_decay": 0.0001},
    )
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(trained_model, metrics, predictions, sep="\n")

    # save model to a file using torch.save
    model.save("/tmp/model_object.pt")
