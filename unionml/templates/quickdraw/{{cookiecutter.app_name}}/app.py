from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from quickdraw.dataset import QuickDrawDataset, get_quickdraw_class_names
from quickdraw.model import init_model, quickdraw_compute_metrics, train_quickdraw
from transformers import EvalPrediction

from unionml import Dataset, Model

dataset = Dataset(name="quickdraw_dataset", test_size=0.2, shuffle=True)
model = Model(name="quickdraw_classifier", init=init_model, dataset=dataset)


@dataset.reader(cache=True, cache_version="1")
def reader(data_dir: str, max_examples_per_class: int = 1000, class_limit: int = 5) -> QuickDrawDataset:
    return QuickDrawDataset(data_dir, max_examples_per_class, class_limit=class_limit)


@dataset.splitter
def splitter(
    quickdraw_data: QuickDrawDataset,
    test_size: float,
    **kwargs,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    return quickdraw_data.split(pct=test_size)


@model.trainer(cache=True, cache_version="1")
def trainer(
    module: nn.Module,
    quickdraw_dataset: torch.utils.data.Subset,
    *,
    num_epochs: int = 20,
    batch_size: int = 256,
) -> nn.Module:
    return train_quickdraw(module, quickdraw_dataset, num_epochs, batch_size)


@model.evaluator
def evaluator(module: nn.Module, quickdraw_dataset: QuickDrawDataset) -> float:
    cuda = torch.cuda.is_available()
    module = module.cuda() if cuda else module
    acc = []
    for features, label_ids in torch.utils.data.DataLoader(quickdraw_dataset, batch_size=256):
        features = features.to("cuda") if cuda else features
        label_ids = label_ids.to("cuda") if cuda else label_ids
        metrics = quickdraw_compute_metrics(EvalPrediction(module(features), label_ids))
        acc.append(metrics["acc1"])
    module.cpu()
    return float(sum(acc) / len(acc))


@dataset.feature_loader
def feature_loader(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0


@model.predictor(cache=True, cache_version="1")
def predictor(module: nn.Module, features: torch.Tensor) -> dict:
    module.eval()
    if torch.cuda.is_available():
        module, features = module.cuda(), features.cuda()
    with torch.no_grad():
        probabilities = nn.functional.softmax(module(features)[0], dim=0)
    class_names = get_quickdraw_class_names()
    values, indices = torch.topk(probabilities, 3)
    return {class_names[i]: v.item() for i, v in zip(indices, values)}


# attach Flyte demo cluster metadata
model.remote(
    dockerfile="Dockerfile",
    config_file=str(Path.home() / ".flyte" / "config.yaml"),
    project="{{ cookiecutter.project_name }}",
    domain="development",
)

# serve with FastAPI
app = FastAPI()
model.serve(app)


if __name__ == "__main__":

    data_dir = "/tmp/quickdraw_data"
    num_classes = 10  # max number of classes is 345
    max_examples_per_class = 500
    num_epochs = 1
    batch_size = 256

    model_object, metrics = model.train(
        hyperparameters={"num_classes": num_classes},
        trainer_kwargs={"num_epochs": num_epochs, "batch_size": batch_size},
        data_dir=data_dir,
        max_examples_per_class=max_examples_per_class,
        class_limit=num_classes,
    )
    quickdraw_dataset = QuickDrawDataset(data_dir, max_examples_per_class, class_limit=num_classes)

    features, _ = quickdraw_dataset[0]
    prediction = model.predict(features.squeeze(0))
    print(model_object, metrics, prediction, sep="\n")

    # save model to a file, using joblib as the default serialization format
    model.save("/tmp/model_object.joblib")
