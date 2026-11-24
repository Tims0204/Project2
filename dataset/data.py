import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from datasets import load_dataset

def load_data():
    # the old convert to float is unnessesary and costs performance
    ds = load_dataset("uoft-cs/cifar10")
    def to_np(split):
        x = np.stack([np.array(img.convert("RGB")) for img in split["img"]])  # uint8, NHWC
        return x, np.array(split["label"])
    xtr, ytr = to_np(ds["train"]); xte, yte = to_np(ds["test"])
    return xtr, ytr, xte, yte



class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]                    # uint8 Tensor [3,32,32]
        if self.transform: 
            x = self.transform(x)
        return x, self.labels[idx]              # uint8 
