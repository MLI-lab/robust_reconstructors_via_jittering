from __future__ import print_function
import torch
import numpy as np
from torch.utils.data import Dataset
import math

class UcSubspaceDataset(Dataset):

    def __init__(self, n, d, size, device):
        self.n = n
        self.d = d
        self.size = size
        self.device = device

        mUt = torch.eye(n, device=device)[:d,:] # size (d, n)
        subspace_samples = torch.randn( (size, d), device=device)
        self.tensor_examples = torch.matmul(subspace_samples, mUt)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.tensor_examples[idx, ...], None