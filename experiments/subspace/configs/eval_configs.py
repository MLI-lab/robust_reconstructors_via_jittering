import os
from experiments.subspace.configs.train_configs import train_step_configs

from generic.step import ExperimentStep
from experiments.subspace.steps.eval.eval import evaluate

from experiments.subspace.configs.base_config import (
    base_artifact_path__default
)

from experiments.subspace.configs.preprocess_configs import (
    preprocessing_config_test,
    preprocessing_config_linear_decay_test
)

base_eval_dir = "eval_data"
step_name = "eval"

evaluate_adversarial_subspace_denoising = {
    "base_dir"                : train_step_configs["adversarial_training_denoising_subspace"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.2],
    "adv_its"               : 100,
    "batch_size"            : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_adversarial_subspace_denoising")
}

evaluate_jittering_subspace_denoising = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_denoising_subspace"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.2],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 100,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_jittering_subspace_denoising")
}
evaluate_standard_subspace_denoising = {
    "base_dir"              : train_step_configs["standard_training_denoising_subspace"].artifact_path,
    "test_dataset_config"     : preprocessing_config_test,
    "fixed_noise_levels"    : [0.2],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 100,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_standard_subspace_denoising")
}

evaluate_adversarial_subspace_linear_decay = {
    "base_dir"                : train_step_configs["adversarial_training_linear_decay_subspace"].artifact_path,
    "test_dataset_config"     : preprocessing_config_linear_decay_test,
    "fixed_noise_levels"    : [0.2],
    "adv_its"               : 100,
    "batch_size"            : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_adversarial_subspace_linear_decay")
}

evaluate_jittering_subspace_linear_decay = {
    "base_dir"              : train_step_configs["jittering_via_hpsearch_training_linear_decay_subspace"].artifact_path,
    "test_dataset_config"     : preprocessing_config_linear_decay_test,
    "fixed_noise_levels"    : [0.2],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 100,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_jittering_subspace_linear_decay")
}
evaluate_standard_subspace_linear_decay = {
    "base_dir"              : train_step_configs["standard_training_linear_decay_subspace"].artifact_path,
    "test_dataset_config"     : preprocessing_config_linear_decay_test,
    "fixed_noise_levels"    : [0.2],
    "adv_its"               : 100,
    "adv_test"              : True,
    "std_test"              : True,
    "batch_size"            : 100,
    "df_save_dir"           : os.path.join(base_artifact_path__default, step_name, "evaluate_standard_subspace_linear_decay")
}

eval_step_configs = {
    "evaluate_adversarial_subspace_denoising" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_adversarial_subspace_denoising["df_save_dir"],
        parameters=evaluate_adversarial_subspace_denoising
    ),
    "evaluate_jittering_subspace_denoising" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_jittering_subspace_denoising["df_save_dir"],
        parameters=evaluate_jittering_subspace_denoising
    ),
    "evaluate_standard_subspace_denoising" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_standard_subspace_denoising["df_save_dir"],
        parameters=evaluate_standard_subspace_denoising
    ),
    "evaluate_adversarial_subspace_linear_decay" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_adversarial_subspace_linear_decay["df_save_dir"],
        parameters=evaluate_adversarial_subspace_linear_decay
    ),
    "evaluate_jittering_subspace_linear_decay" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_jittering_subspace_linear_decay["df_save_dir"],
        parameters=evaluate_jittering_subspace_linear_decay
    ),
    "evaluate_standard_subspace_linear_decay" : ExperimentStep(
        step_func=evaluate,
        artifact_path=evaluate_standard_subspace_linear_decay["df_save_dir"],
        parameters=evaluate_standard_subspace_linear_decay
    )
}