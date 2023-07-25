import os
from experiments.natural_images.configs.train_configs import train_step_configs

from experiments.natural_images.configs.base_config import (
    base_dataset_path,
    base_artifact_path__default
)

from experiments.natural_images.configs.preprocess_configs import (
    preprocessing_config_test
)

from generic.step import ExperimentStep
from experiments.natural_images.steps.visualize import visualize_network_reconstructions

step_name = "visualize"

visualize_UNet_Adversarial025SigmaRGB_denoising = {
    "base_dir"                : train_step_configs["adversarial_training_denoising_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.06,
    "zeta_test"             : 0.06,
    "noise_level"           : 0.25,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_Adversarial025SigmaRGB_denoising")
}

visualize_UNet_Jittering025SigmaRGB_denoising = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_denoising_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.06,
    "zeta_test"             : 0.06,
    "noise_level"           : 0.25,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_Jittering025SigmaRGB_denoising")
}
visualize_UNet_025SigmaStandardRGB_denoising = {
    "base_dir"              : train_step_configs["standard_training_denoising_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.00,
    "zeta_test"             : 0.06,
    "noise_level"           : 0.25,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_025SigmaStandardRGB_denoising")
}

visualize_UNet_Adversarial025SigmaRGB_gaussian8x8 = {
    "base_dir"              : train_step_configs["adversarial_training_gaussian8x8_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.06,
    "zeta_test"             : 0.06,
    "noise_level"           : 0.25,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_Adversarial025SigmaRGB_gaussian8x8")
}

visualize_UNet_Jittering025SigmaRGB_gaussian8x8 = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_gaussian8x8_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.06,
    "zeta_test"             : 0.06,
    "noise_level"           : 0.25,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_Jittering025SigmaRGB_gaussian8x8")
}
visualize_UNet_025SigmaStandardRGB_gaussian8x8 = {
    "base_dir"              : train_step_configs["standard_training_gaussian8x8_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.00,
    "zeta_test"             : 0.06,
    "noise_level"           : 0.25,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_025SigmaStandardRGB_gaussian8x8")
}

step_configs = {
    "visualize_UNet_Adversarial025SigmaRGB_denoising" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_Adversarial025SigmaRGB_denoising["output_dir"],
        parameters=visualize_UNet_Adversarial025SigmaRGB_denoising
    ),
    "visualize_UNet_Jittering025SigmaRGB_denoising" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_Jittering025SigmaRGB_denoising["output_dir"],
        parameters=visualize_UNet_Jittering025SigmaRGB_denoising
    ),
    "visualize_UNet_025SigmaStandardRGB_denoising" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_025SigmaStandardRGB_denoising["output_dir"],
        parameters=visualize_UNet_025SigmaStandardRGB_denoising
    ),
    "visualize_UNet_Adversarial025SigmaRGB_gaussian8x8" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_Adversarial025SigmaRGB_gaussian8x8["output_dir"],
        parameters=visualize_UNet_Adversarial025SigmaRGB_gaussian8x8
    ),
    "visualize_UNet_Jittering025SigmaRGB_gaussian8x8" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_Jittering025SigmaRGB_gaussian8x8["output_dir"],
        parameters=visualize_UNet_Jittering025SigmaRGB_gaussian8x8
    ),
    "visualize_UNet_025SigmaStandardRGB_gaussian8x8" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_025SigmaStandardRGB_gaussian8x8["output_dir"],
        parameters=visualize_UNet_025SigmaStandardRGB_gaussian8x8
    )
}