import math
import urllib
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import torch
from tqdm import tqdm

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
