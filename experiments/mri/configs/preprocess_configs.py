from experiments.mri.configs.base_config import (
    base_dataset_path_train,
    base_dataset_path_val
)

preprocessing_config_train = {
    "path" : base_dataset_path_train,
    "filtered" : False,
    "filtering_csv" : "",
    "subset_factor" : 1.0,
    "split_factor" : 0.9,
    "deterministic_split" : True,
    "first_part" : True
}

preprocessing_config_val = {
    "path" : base_dataset_path_train,
    "filtered" : False,
    "filtering_csv" : "",
    "subset_factor" : 1.0,
    "split_factor" : 0.9,
    "deterministic_split" : True,
    "first_part" : False
}

preprocessing_config_test = {
    "path" : base_dataset_path_val,
    "filtered" : False,
    "filtering_csv" : "",
    "subset_factor" : 1.0,
    "split_factor" : 1.0,
    "deterministic_split" : False,
    "first_part" : True
}