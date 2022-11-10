---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# QuickDraw: A Pictionary App

+++ {"tags": ["add-colab-badge"]}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unionai-oss/unionml/blob/main/docs/notebooks/quickdraw.ipynb)

+++

In this example, we'll see how to create pictionary app that uses the
[QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset
to train a convolutional neural net to predict the semantic label of a
hand-drawn picture.

We'll break this tutorial up into two parts:

1. Creating plain Python classes and functions to implement the quickdraw dataset
   and model using [`pytorch`](https://pytorch.org/) and the Hugging Face
   [`transformers`](https://huggingface.co/docs/transformers/index) library.
2. Using the pieces in part 1 to create a UnionML app for training a model
   and serving predictions using a [`gradio`](https://gradio.app/) widget.

## Part 1: Implementing the Quickdraw Model

```{note}
This tutorial is adapted from this [gradio guide](https://gradio.app/building_a_pictionary_app/),
and you can find the original notebook [here](https://github.com/nateraw/quickdraw-pytorch).
```

```{code-cell}
:tags: [remove-cell]

%%capture
!pip install 'gradio<=3.0.10' numpy tqdm requests torch transformers unionml
```

+++ {"tags": ["remove-cell"]}

> If you're running this notebook in google colab, you need to restart the kernel to
> make sure that the newly installed packages are correctly imported in the next line below.

+++

First let's import everything we need:

```{code-cell}
import math
from typing import List, Optional

import urllib.request
from tqdm.auto import tqdm
from pathlib import Path
import requests
import torch
import numpy as np
```

Then let's implement some helper functions for downloading the quickdraw data and loading it
into memory:

```{code-cell}
CLASSES_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
DATASET_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def get_quickdraw_class_names():
    """Get the class names associated with the quickdraw dataset."""
    return [*sorted(x.replace(' ', '_') for x in requests.get(CLASSES_URL).text.splitlines())]


def download_quickdraw_dataset(
    root: str = "./data",
    limit: Optional[int] = None,
    class_names: List[str]=None,
):
    """Download quickdraw data to a directory containing files for each class label."""
    class_names = class_names or get_quickdraw_class_names()
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    print("Downloading Quickdraw Dataset...")
    for class_name in tqdm(class_names[:limit]):
        urllib.request.urlretrieve(
            f"{DATASET_URL}{class_name.replace('_', '%20')}.npy",
            root / f"{class_name}.npy"
        )


def load_quickdraw_data(root: str = "./data", max_items_per_class: int = 5000):
    """Load quickdraw data in to memory, returning features, labels, and class names."""
    x = np.empty([0, 784], dtype=np.uint8)
    y = np.empty([0], dtype=np.int64)
    class_names = []
    print(f"Loading {max_items_per_class} examples for each class from the Quickdraw Dataset...")
    for idx, file in enumerate(tqdm(sorted(Path(root).glob('*.npy')))):
        data = np.load(file, mmap_mode='r')[0: max_items_per_class, :]
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, np.full(data.shape[0], idx))
        class_names.append(file.stem)
    return x, y, class_names
```

### QuickDraw Dataset

Next we implement the `QuickDrawDataset` using `torch.utils.data.Dataset`:

```{code-cell}
class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_items_per_class=5000, class_limit=None):
        super().__init__()
        download_quickdraw_dataset(root, class_limit)
        self.X, self.Y, self.classes = load_quickdraw_data(root, max_items_per_class)

    def __getitem__(self, idx):
        x = (self.X[idx] / 255.).astype(np.float32).reshape(1, 28, 28)
        y = self.Y[idx]
        return torch.from_numpy(x), y.item()

    def __len__(self):
        return len(self.X)

    @staticmethod
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([item[0] for item in batch]),
            'labels': torch.LongTensor([item[1] for item in batch]),
        }

    def split(self, pct=0.1):
        indices = torch.randperm(len(self)).tolist()
        n_val = math.floor(len(indices) * pct)
        train_ds = torch.utils.data.Subset(self, indices[:-n_val])
        val_ds = torch.utils.data.Subset(self, indices[-n_val:])
        return train_ds, val_ds
```

As you'll see later, this class is important so that the `transformers` library can
handle the automatic batching of data during training.

### QuickDraw Model and Trainer

Now let's define the model architecture for our ConvNet:

```{code-cell}
from torch import nn

def init_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2304, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )
```

As you can see it's a fairly straightforward 2D ConvNet architecture that uses a square
kernel size of 3, Relu layers for its non-linear activation operator, and max-pooling.

Next, let's create a subclass of `transformers.Trainer` to implement a custom loss function:

```{code-cell}
from transformers import EvalPrediction, Trainer, TrainingArguments
from transformers.modeling_utils import ModelOutput


class QuickDrawTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        logits, labels = model(inputs["pixel_values"]), inputs.get("labels")
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, ModelOutput(logits=logits, loss=loss)) if return_outputs else loss
```

Then, let's define helper functions to compute the accuracy metric, which will be how we'll
judge the performance of our model:

```{code-cell}
# Adapted from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def quickdraw_compute_metrics(p: EvalPrediction):
    if p.label_ids is None:
        return {}
    acc1, acc5 = accuracy(p.predictions, p.label_ids, topk=(1, 5))
    return {'acc1': acc1, 'acc5': acc5}
```

Finally, let's create a `train_quickdraw` function that will serve as the main entrypoint
for training:

```{code-cell}
from datetime import datetime

def train_quickdraw(module: nn.Module, dataset: QuickDrawDataset, num_epochs: int, batch_size: int):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    training_args = TrainingArguments(
        output_dir=f'~/.tmp/outputs_20k_{timestamp}',
        save_strategy='epoch',
        report_to=['tensorboard'],
        logging_strategy='steps',
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
```

Why did we go through all of this trouble of implementing the dataset and model classes/functions
instead of embedding it inside our UnionML app?

Well, it often makes sense to separate the concerns of the dataset/model implementation from the
application code that will scale or serve it, especially for more complex projects. Depending on the
the complexity of the data processing and modeling logic needed to train your model, you may want to
create separate functions/classes/modules to abstract it away.

In the next section, we'll see that this pays dividends in terms of readability and maintainability.

## Part 2: Creating a UnionML Pictionary App

Now that we have all the pieces we need to train our model, let's create the UnionML app. First we
import what we need and define our `unionml.Dataset` and `unionml.Model` objects:

```{code-cell}
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from transformers import EvalPrediction
from unionml import Dataset, Model

dataset = Dataset(name="quickdraw_dataset", test_size=0.2, shuffle=True)
model = Model(name="quickdraw_classifier", init=init_model, dataset=dataset)
```

### Reading the Dataset

Then, we implement the `reader` function, which returns a `QuickDrawDataset`:

```{code-cell}
@dataset.reader(cache=True, cache_version="1")
def reader(
    data_dir: str, max_examples_per_class: int = 1000, class_limit: int = 5
) -> QuickDrawDataset:
    return QuickDrawDataset(data_dir, max_examples_per_class, class_limit=class_limit)
```

### Training

Next, we define the `trainer` function, using the `quickdraw_trainer` helper function we
defined above and an `evaluator` function to let UnionML know how to evaluate the model
on some partition of the data:

```{code-cell}
@model.trainer(cache=True, cache_version="1")
def trainer(
    module: nn.Module,
    dataset: QuickDrawDataset,
    *,
    num_epochs: int = 20,
    batch_size: int = 256,
) -> nn.Module:
    return train_quickdraw(module, dataset, num_epochs, batch_size)

@model.evaluator
def evaluator(module: nn.Module, dataset: QuickDrawDataset) -> float:
    cuda = torch.cuda.is_available()
    module = module.cuda() if cuda else module
    acc = []
    for features, label_ids in torch.utils.data.DataLoader(dataset, batch_size=256):
        features = features.to("cuda") if cuda else features
        label_ids = label_ids.to("cuda") if cuda else label_ids
        metrics = quickdraw_compute_metrics(EvalPrediction(module(features), label_ids))
        acc.append(metrics["acc1"])
    module.cpu()
    return float(sum(acc) / len(acc))
```

### Prediction

Because we expect to generate predictions from raw images in the form of a numpy array,
we need to register a `feature_loader` function in the `dataset` object:

```{code-cell}
@dataset.feature_loader
def feature_loader(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
```

Then we can define a `predictor` function that consumes the output of `feature_loader`:

```{code-cell}
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
```

### Training a Model Locally

Awesome! If you've been following along in your editor or a Jupyter notebook, you just implemented
a pictionary app in UnionML ⭐️

Now let's train a model just using 10 classes, with 500 examples per class, for 1 epoch. This
model won't perform that well, so feel free to change these numbers up in the code below:

```{code-cell}
num_classes = 10  # max number of classes is 345
max_examples_per_class = 500
num_epochs = 1
batch_size = 256

model.train(
    hyperparameters={"num_classes": num_classes},
    trainer_kwargs={"num_epochs": num_epochs, "batch_size": batch_size},
    data_dir="/tmp/quickdraw_data",
    max_examples_per_class=max_examples_per_class,
    class_limit=num_classes,
)
```

### Serving on a Gradio Widget

And now the moment of truth 🙌

To create a `gradio` widget, we can simply use the `model.predict` method into the
`gradio.Interface` object using a `lambda` function to handle the `None` case when we press
the `clear` button on the widget:

```{code-cell}
:tags: [remove-output]

import gradio as gr

gr.Interface(
    fn=lambda img: img if img is None else model.predict(img),
    inputs="sketchpad",
    outputs="label",
    live=True,
    allow_flagging="never",
).launch()
```

You might notice that the model may not perform as well as you might expect...
welcome to the world of machine learning practice! To obtain a better model given
a fixed dataset, feel free to play around with the model hyperparameters or even
switch up the model type/architecture that's defined in the `trainer` function.
