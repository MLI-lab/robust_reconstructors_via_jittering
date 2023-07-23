import os
from experiments.mri.configs.train_configs import train_step_configs

from experiments.mri.configs.base_config import (
    base_artifact_path__default
)

from experiments.mri.configs.preprocess_configs import (
    preprocessing_config_test
)

from generic.step import ExperimentStep
from experiments.mri.steps.visualize import visualize_network_reconstructions

step_name = "visualize"

visualize_UNet_adversarial_mri = {
    "base_dir"                : train_step_configs["adversarial_training_mri"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.03,
    "zeta_test"             : 0.03,
    "noise_level"           : 0.0,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_adversarial_mri")
}

visualize_UNet_jittering_mri = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_mri"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.03,
    "zeta_test"             : 0.03,
    "noise_level"           : 0.0,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_jittering_mri")
}
visualize_UNet_standard_mri = {
    "base_dir"              : train_step_configs["standard_training_mri"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "test_dataset_config"     : preprocessing_config_test,
    "adv_its"               : 100,
    "adv_random_start"      : False,
    "zeta_train"            : 0.00,
    "zeta_test"             : 0.03,
    "noise_level"           : 0.0,
    "data_indices"          : [0, 1, 2, 3, 4],
    "output_dir"           : os.path.join(base_artifact_path__default, step_name, "visualize_UNet_standard_mri")
}


step_configs = {
    "visualize_UNet_adversarial_mri" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_adversarial_mri["output_dir"],
        parameters=visualize_UNet_adversarial_mri
    ),
    "visualize_UNet_jittering_mri" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_jittering_mri["output_dir"],
        parameters=visualize_UNet_jittering_mri
    ),
    "visualize_UNet_standard_mri" : ExperimentStep(
        step_func=visualize_network_reconstructions,
        artifact_path=visualize_UNet_standard_mri["output_dir"],
        parameters=visualize_UNet_standard_mri
    )
}