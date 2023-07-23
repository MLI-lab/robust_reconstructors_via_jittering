# %%
from __future__ import print_function
import torch
from torch.utils.data import Dataset
import tqdm
import os
import glob

class StackedTensorOnDeviceDataset(Dataset):
    def __init__(self, base_dataset, device):
        self.tensor_list_x = torch.zeros( (len(base_dataset), *base_dataset[0][0].shape), device=device)
        self.tensor_list_y = torch.zeros( (len(base_dataset), *base_dataset[0][1].shape), device=device)
        for i in tqdm.trange(len(base_dataset), position=0):
            x,y = base_dataset[i]
            self.tensor_list_x[i,...] = x
            self.tensor_list_y[i,...] = y

    def __len__(self):
        return len(self.tensor_list_x)

    def __getitem__(self, idx):
        return self.tensor_list_x[idx, ...], self.tensor_list_y[idx, ...]

class TensorListDataset(Dataset):
    def __init__(self, path, transform=None, device=None, single_tensor=True):
        self.transform = transform
        self.single_tensor = single_tensor
        if single_tensor:
            self.tensor_list = torch.load(glob.glob(os.path.join(path, "*.pt"))[0])
            if device != None:
                self.tensor_list = self.tensor_list.to(device)
        else:
            if device is None:
                device = "cpu"
            self.tensor_list = [torch.load(tensor_filepath).to(device) for tensor_filepath in glob.glob(os.path.join(path, "*.pt"))]

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        x = self.tensor_list[idx]
        xr = self.transform(x) if self.transform != None else x
        return xr, None