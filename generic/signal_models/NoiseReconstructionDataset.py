# %%
from __future__ import print_function
import numpy as np
from torch.utils.data import Dataset

# %%
class NoiseReconstructionDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, norm_filter_pseudo_len = None, norm_filter_center=None, norm_filter_width=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.norm_filter_center = norm_filter_center
        self.norm_filter_width = norm_filter_width
        self.norm_filter_pseudo_len = norm_filter_pseudo_len
        self.norm_filter_offset = 0

    def __len__(self):
        return len(self.dataset) if self.norm_filter_pseudo_len is None else self.norm_filter_pseudo_len

    def _get_img(self, idx):
        if self.norm_filter_center != None and self.norm_filter_width != None:
            l = len(self.dataset)
            offset = 0
            while offset < l:
                x, _ = self.dataset[ (idx+offset) % l]
                l2norm_square = np.inner(x.flatten(), x.flatten())
                if (l2norm_square > self.norm_filter_center - self.norm_filter_width) and (l2norm_square < self.norm_filter_center + self.norm_filter_width):
                    self.norm_filter_offset = (self.norm_filter_offset + offset) % l
                    return x
                offset+=1
        else:
            x, _ = self.dataset[idx]
            return x


    def __getitem__(self, idx):
        x = self._get_img(idx)
        xr = self.transform(x) if self.transform != None else x
        yr = self.target_transform(x) if self.target_transform != None else x
        return xr, yr