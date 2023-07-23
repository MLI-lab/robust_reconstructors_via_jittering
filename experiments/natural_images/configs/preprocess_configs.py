from experiments.natural_images.configs.base_config import (
    base_dataset_path,
    base_artifact_path__datasets
)

step_name = "preprocess"
rgb_normalization_none = {"enabled" : False}
rgb_normalization = {"enabled" : True, "mean" : [0.485, 0.456, 0.406], "std" : [0.229, 0.224, 0.225]}
grayscale_normalization = {"enabled" : True, "mean" : [0.454], "std" : [0.199] }

preprocessing_config_train = {
    "path_from" : f"{base_dataset_path}/train",
    "path_to"   : f"{base_artifact_path__datasets}/{step_name}/preprocessing_train",
    "file_glob_append" : "*/*",
    "subset_settings" : {"fraction_start" : 0, "fraction_stop" : 1.0, "number_samples" : 34000, "random_subset" : True, "random_seed" : 0},
    "transform_normalize" : rgb_normalization,
    "transform_random_crop" : {"enabled" : True, "width" : 128, "height" : 128},
    "transform_grayscale" : False,
    "output_format" : {"tensors" : True, "single_tensor" : True},
    "filter" : {"enabled" : False, "reps" : 1, "center": 1000, "width": 500}
}

preprocessing_config_val = {
    "path_from" : f"{base_dataset_path}/val",
    "path_to"   : f"{base_artifact_path__datasets}/{step_name}/preprocessing_val",
    "file_glob_append" : "*/*",
    "subset_settings" : {"fraction_start" : 0.0, "fraction_stop" : 1.0, "number_samples" : 4000, "random_subset" : True, "random_seed" : 0},
    "transform_normalize" : rgb_normalization,
    "transform_random_crop" : {"enabled" : True, "width" : 128, "height" : 128},
    "transform_grayscale" : False,
    "output_format" : {"tensors" : True, "single_tensor" : True},
    "filter" : {"enabled" : False, "reps" : 1, "center": 1000, "width": 500}
}
preprocessing_config_test = {
    "path_from" : f"{base_dataset_path}/val",
    "path_to"   : f"{base_artifact_path__datasets}/{step_name}/preprocessing_test",
    "file_glob_append" : "*/*",
    "subset_settings" : {"fraction_start" : 0.0, "fraction_stop" : 1.0, "number_samples" : 2000, "random_subset" : True, "random_seed" : 0},
    "transform_normalize" : rgb_normalization,
    "transform_random_crop" : {"enabled" : False},
    "transform_grayscale" : False,
    "output_format" : {"tensors" : True, "single_tensor" : False},
    "filter" : {"enabled" : False, "reps" : 1, "center": 1000, "width": 500}
}

preprocessing_config_test_visual = {
    "path_from" : f"{base_dataset_path}/test",
    "path_to"   : f"{base_artifact_path__datasets}/{step_name}/preprocessing_test_visual",
    "file_glob_append" : "*",
    "subset_settings" : {"fraction_start" : 0.0, "fraction_stop" : 1.0, "number_samples" : 5, "random_subset" : True, "random_seed" : 0},
    "transform_normalize" : rgb_normalization,
    "transform_random_crop" : {"enabled" : False},
    "transform_grayscale" : False,
    "output_format" : {"tensors" : False, "single_tensor" : False},
    "filter" : {"enabled" : False, "reps" : 1, "center": 1000, "width": 500}
}

from generic.step import ExperimentStep
from experiments.natural_images.steps.preprocessing import generate_dataset_single

step_configs = {
    "preprocessing_config_train" : ExperimentStep(
        step_func=generate_dataset_single,
        artifact_path=preprocessing_config_train["path_to"],
        parameters = preprocessing_config_train
    ),
    "preprocessing_config_val" : ExperimentStep(
        step_func=generate_dataset_single,
        artifact_path=preprocessing_config_val["path_to"],
        parameters = preprocessing_config_val
    ),
    "preprocessing_config_test" : ExperimentStep(
        step_func=generate_dataset_single,
        artifact_path=preprocessing_config_test["path_to"],
        parameters = preprocessing_config_test
    ),
    "preprocessing_config_test_visual" : ExperimentStep(
        step_func=generate_dataset_single,
        artifact_path=preprocessing_config_test_visual["path_to"],
        parameters = preprocessing_config_test_visual
    )
}