# %%
from __future__ import print_function
import numpy as np
from torch.utils.data import Dataset

class ReconstructionDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        xr = self.transform(x) if self.transform != None else x
        yr = self.target_transform(x) if self.target_transform != None else x
        return xr, yr