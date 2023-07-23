import os
from experiments.mri.configs.train_configs import train_step_configs
from experiments.mri.configs.base_config import base_artifact_path__default
from generic.step import ExperimentStep
from experiments.mri.steps.eval.eval import evaluate
from experiments.mri.configs.preprocess_configs import (
    preprocessing_config_test
)

base_eval_dir = "eval_data"
step_name = "eval"

evaluate_UNet_adversarial_mri = {
    "base_dir"                : train_step_configs["adversarial_training_mri"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.0],
    "adv_its"               : 100,
    "batch_size"            : 1,
    "adv_test"              : True,
    "std_test"              : True,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_adversarial_mri")
}

evaluate_UNet_jittering_mri = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_mri"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.0],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 1,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_jittering_mri")
}
evaluate_UNet_standard_mri = {
    "base_dir"              : train_step_configs["standard_training_mri"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.0],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 1,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_standard_mri")
}

eval_step_configs = {
    "evaluate_UNet_adversarial_mri" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_adversarial_mri["df_save_dir"],
        parameters=evaluate_UNet_adversarial_mri
    ),
    "evaluate_UNet_jittering_mri" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_jittering_mri["df_save_dir"],
        parameters=evaluate_UNet_jittering_mri
    ),
    "evaluate_UNet_standard_mri" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_standard_mri["df_save_dir"],
        parameters=evaluate_UNet_standard_mri
    )
}