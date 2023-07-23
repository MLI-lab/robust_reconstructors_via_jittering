import os
from experiments.natural_images.configs.train_configs import train_step_configs

from generic.step import ExperimentStep
from experiments.natural_images.steps.eval.eval import evaluate

from experiments.natural_images.configs.base_config import (
    base_artifact_path__default
)

from experiments.natural_images.configs.preprocess_configs import (
    preprocessing_config_test
)

base_eval_dir = "eval_data"
step_name = "eval"

evaluate_UNet_Adversarial025SigmaRGB_denoising = {
    "base_dir"                : train_step_configs["adversarial_training_denoising_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.25],
    "adv_its"               : 100,
    "batch_size"            : 1,
    "adv_test"              : True,
    "std_test"              : True,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_Adversarial025SigmaRGB_denoising")
}

evaluate_UNet_Jittering025SigmaRGB_denoising = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_denoising_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.25],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 1,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_evaluate_UNet_Jittering025SigmaRGB_denoising")
}
evaluate_UNet_025SigmaStandardRGB_denoising = {
    "base_dir"              : train_step_configs["standard_training_denoising_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.25],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 1,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_025SigmaStandardRGB_denoising")
}

evaluate_UNet_Adversarial025SigmaRGB_gaussian8x8 = {
    "base_dir"              : train_step_configs["adversarial_training_denoising_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.25],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_Adversarial025SigmaRGB_gaussian8x8")
}

evaluate_UNet_Jittering025SigmaRGB_gaussian8x8 = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_gaussian8x8_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.25],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_Jittering025SigmaRGB_gaussian8x8")
}
evaluate_UNet_025SigmaStandardRGB_gaussian8x8 = {
    "base_dir"              : train_step_configs["standard_training_gaussian8x8_rgb"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.25],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_UNet_025SigmaStandardRGB_gaussian8x8")
}

eval_step_configs = {
    "evaluate_UNet_Adversarial025SigmaRGB_denoising" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_Adversarial025SigmaRGB_denoising["df_save_dir"],
        parameters=evaluate_UNet_Adversarial025SigmaRGB_denoising
    ),
    "evaluate_UNet_Jittering025SigmaRGB_denoising" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_Jittering025SigmaRGB_denoising["df_save_dir"],
        parameters=evaluate_UNet_Jittering025SigmaRGB_denoising
    ),
    "evaluate_UNet_025SigmaStandardRGB_denoising" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_025SigmaStandardRGB_denoising["df_save_dir"],
        parameters=evaluate_UNet_025SigmaStandardRGB_denoising
    ),
    "evaluate_UNet_Adversarial025SigmaRGB_gaussian8x8" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_Adversarial025SigmaRGB_gaussian8x8["df_save_dir"],
        parameters=evaluate_UNet_Adversarial025SigmaRGB_gaussian8x8
    ),
    "evaluate_UNet_Jittering025SigmaRGB_gaussian8x8" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_Jittering025SigmaRGB_gaussian8x8["df_save_dir"],
        parameters=evaluate_UNet_Jittering025SigmaRGB_gaussian8x8
    ),
    "evaluate_UNet_025SigmaStandardRGB_gaussian8x8" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_UNet_025SigmaStandardRGB_gaussian8x8["df_save_dir"],
        parameters=evaluate_UNet_025SigmaStandardRGB_gaussian8x8
    )
}