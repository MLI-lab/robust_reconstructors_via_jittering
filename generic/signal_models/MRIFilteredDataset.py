from generic.signal_models.MRISliceDataset import MRISliceDataset
from generic.signal_models.MRITransformedDataset import MRITransformedDataset

import torch
import pandas as pd

def create_filtered_mri_train_test_datasets(datapath, challenge, transform=None, attribute_df_path="knee_trainset.csv", train_dataset_fraction = 0.3, train_split_fraction = 0.9, train_test_split_random = False, encodeX = 640, encodeY = 368):
    df_dataset_tr = pd.read_csv(attribute_df_path)  
    df_filtered = df_dataset_tr[(df_dataset_tr['encodeX']==encodeX) & (df_dataset_tr['encodeY']==encodeY)]
    files_tr = list(datapath + df_filtered['filename'])
    dataset_train_full_base = MRISliceDataset(files_tr, challenge=challenge, transform=transform)

    dataset_train_subset_size = int(train_dataset_fraction*len(dataset_train_full_base))
    train_size = int(train_split_fraction*dataset_train_subset_size)

    if train_test_split_random:
        dataset_train_base, dataset_test_base = torch.utils.data.random_split(dataset_train_full_base, [train_size, dataset_train_subset_size-train_size])
    else:
        dataset_train_base = torch.utils.data.Subset(dataset_train_full_base, range(train_size))
        dataset_test_base = torch.utils.data.Subset(dataset_train_full_base, range(train_size, dataset_train_subset_size))

    return MRITransformedDataset(dataset_train_base), MRITransformedDataset(dataset_test_base)