import os
import tqdm
from glob import glob
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision
from PIL import Image
import math
import random

from fastmri.data.mri_data import SliceDataset

from torch.utils.tensorboard import SummaryWriter
from generic.helper.device_helpers import get_last_gpu__or__cpu
from generic.helper.tb_helpers import pltfig_to_tensor
from generic.reconstruction_models.UNetReconstruction import UNet
from generic.reconstruction_models.loss_functions import get_loss_fn
from generic.robust_training.AttackerModel import AttackerModel

from generic.signal_models.MRISliceDataset import MRISliceDataset
from generic.signal_models.MRITransformedDataset import MRITransformedDataset

from generic.signal_models.TensorListDataset import TensorListDataset

import pandas as pd

def dataset_via_config(config, device=None):
    """
        Datasets are processed by
        (1) considering a filtered version using a csv (e.g. on measurement resolutions)
        (2) taking a subset deterministically and taking the first part (e.g. first 80%)
        (3) splitting this subset into two parts (either deterministically, or randomly)
        (4) returning one of these two parts
    """
    path_to_fastmri_tr = config["path"]
    filtered = config["filtered"]# if "filtered" in config else False

    if filtered:
        df_dataset_tr = pd.read_csv(config["filtering_csv"]) 
        df_filtered = df_dataset_tr[(df_dataset_tr['encodeX']==640) & (df_dataset_tr['encodeY']==368)]
        # Create list of file paths
        files_tr = list(path_to_fastmri_tr + df_filtered['filename'])
        dataset_full_base = MRISliceDataset(files_tr, challenge='singlecoil', transform=None)
    else:
        dataset_full_base = SliceDataset(root=path_to_fastmri_tr, challenge='singlecoil', transform=None)

    subset_factor = config["subset_factor"] if "subset_factor" in config else 1
    split_factor = config["split_factor"] if "split_factor" in config else 0.9

    dataset_subset_size = int(subset_factor*len(dataset_full_base))

    first_part_size = int(split_factor*dataset_subset_size)
    second_part_size = dataset_subset_size - first_part_size

    if config["deterministic_split"]:
        if config["first_part"]:
            return torch.utils.data.Subset(dataset_full_base, range(first_part_size))
        else:
            return torch.utils.data.Subset(dataset_full_base, range(first_part_size, first_part_size + second_part_size))
    else:
        dataset_subset = torch.utils.data.Subset(dataset_full_base, range(dataset_subset_size))
        datasets = torch.utils.data.random_split(dataset_subset, [first_part_size, second_part_size])
        return datasets[0] if config["first_part"] else datasets[1]