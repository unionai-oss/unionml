from datetime import datetime

import torch
from torch import nn
from transformers import EvalPrediction, Trainer, TrainingArguments
from transformers.modeling_utils import ModelOutput

from .dataset import QuickDrawDataset


class QuickDrawTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        logits, labels = model(inputs["pixel_values"]), inputs.get("labels")
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, ModelOutput(logits=logits, loss=loss)) if return_outputs else loss


def accuracy(output, target, topk=(1,)):
    # Adapted from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
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
        data_collator=QuickDrawDataset.collate_fn,
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
