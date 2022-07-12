"""Helper functions for the quickdraw app."""

import math
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import EvalPrediction, Trainer, TrainingArguments
from transformers.modeling_utils import ModelOutput

CLASSES_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
DATASET_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_items_per_class=5000, class_limit=None):
        super().__init__()
        download_quickdraw_dataset(root, class_limit)
        self.X, self.Y, self.classes = load_quickdraw_data(root, max_items_per_class)

    def __getitem__(self, idx):
        x = (self.X[idx] / 255.0).astype(np.float32).reshape(1, 28, 28)
        y = self.Y[idx]
        return torch.from_numpy(x), y.item()

    def __len__(self):
        return len(self.X)

    @staticmethod
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([item[0] for item in batch]),
            "labels": torch.LongTensor([item[1] for item in batch]),
        }

    def split(self, pct=0.1):
        indices = torch.randperm(len(self)).tolist()
        n_val = math.floor(len(indices) * pct)
        train_ds = torch.utils.data.Subset(self, indices[:-n_val])
        val_ds = torch.utils.data.Subset(self, indices[-n_val:])
        return train_ds, val_ds


def get_quickdraw_class_names():
    """Get the class names associated with the quickdraw dataset."""
    return [*sorted(x.replace(" ", "_") for x in requests.get(CLASSES_URL).text.splitlines())]


def download_quickdraw_dataset(
    root: str = "./data",
    limit: Optional[int] = None,
    class_names: List[str] = None,
):
    """Download quickdraw data to a directory containing files for each class label."""
    class_names = class_names or get_quickdraw_class_names()
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    print("Downloading Quickdraw Dataset...")
    for class_name in tqdm(class_names[:limit]):
        urllib.request.urlretrieve(f"{DATASET_URL}{class_name.replace('_', '%20')}.npy", root / f"{class_name}.npy")


def load_quickdraw_data(root: str = "./data", max_items_per_class: int = 5000):
    """Load quickdraw data in to memory, returning features, labels, and class names."""
    x = np.empty([0, 784], dtype=np.uint8)
    y = np.empty([0], dtype=np.long)
    class_names = []
    print(f"Loading {max_items_per_class} examples for each class from the Quickdraw Dataset...")
    for idx, file in enumerate(tqdm(sorted(Path(root).glob("*.npy")))):
        data = np.load(file, mmap_mode="r")[0:max_items_per_class, :]
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, np.full(data.shape[0], idx))
        class_names.append(file.stem)
    return x, y, class_names


def init_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding="same"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2304, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )


class QuickDrawTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        logits, labels = model(inputs["pixel_values"]), inputs.get("labels")
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, ModelOutput(logits=logits, loss=loss)) if return_outputs else loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def quickdraw_compute_metrics(p: EvalPrediction):
    if p.label_ids is None:
        return {}
    acc1, acc5 = accuracy(p.predictions, p.label_ids, topk=(1, 5))
    return {"acc1": acc1, "acc5": acc5}


def train_quickdraw(module: nn.Module, dataset: QuickDrawDataset, num_epochs: int, batch_size: int):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=f"~/.tmp/outputs_20k_{timestamp}",
        save_strategy="epoch",
        report_to=["tensorboard"],
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=0.003,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        num_train_epochs=num_epochs,
        warmup_steps=10000,
        save_total_limit=5,
    )

    print(f"Training on device: {training_args.device}")

    quickdraw_trainer = QuickDrawTrainer(
        module,
        training_args,
        data_collator=dataset.collate_fn,
        train_dataset=dataset,
        tokenizer=None,
        compute_metrics=quickdraw_compute_metrics,
    )
    train_results = quickdraw_trainer.train()
    quickdraw_trainer.save_model()
    quickdraw_trainer.log_metrics("train", train_results.metrics)
    quickdraw_trainer.save_metrics("train", train_results.metrics)
    quickdraw_trainer.save_state()
    return module
