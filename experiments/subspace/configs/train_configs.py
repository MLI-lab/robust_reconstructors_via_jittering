from experiments.subspace.configs.base_config import (
    base_artifact_path__default
)
from experiments.subspace.configs.hpsearch_configs import (
    hpsearch_denoising_jittering_subspace_eval,
    hpsearch_linear_decay_jittering_subspace_eval
)

import os
step_name = "train"

training_base_config = {
    "epochs":50,
    "zetas_start" : 0,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "train_dataloader_batch_size": 100,
    "test_dataloader_batch_size": 100,
    "num_workers": 0,
    "SGD_lr": 1e-1,
    "SGD_momentum": 0.9,
    "SGD_weight_decay" : 0,
    "adv_its": 10,
    "adv_perturbation_type" : "2",
    "adv_random_start" : False,
    "adv_random_restarts" : False,
    "adv_use_best" : False,
    "adv_random_mode" : "uniform_in_sphere",
    "adv_step_size_factor" : 2.5,
    "unet_classes" : 3,
    "train_eps_factor" : 0, 
    "train_eps_constant" : 0,
    "test_eps_factor" : 0,
    "test_eps_constant" : 0,
    "train_make_adv" : True,
    "test_make_adv" : True,
    "train_pin_memory" : False,
    "test_pin_memory" : False,
    "data_parallel" : False,
    "data_parallel_list" : ["cpu"],
    "tb_image_test_indices" : [10, 40, 80, 99],
    "tb_image_dpi" : 100,
    "tb_subfolder" : "",
    "save_models" : True,
    "save_models_subdir" : "models",
    "fixed_gpu" : False,
    "fixed_gpu_device" : "cuda:1",
    "zeta_by_noise_level" : False,
    "zeta_by_wd" : False,
    "train_optimizer_name" : "Adam",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_args" : {"gamma" : 0.1, "step_size" : 25}
}

from experiments.subspace.configs.preprocess_configs import (
    preprocessing_config_train,
    preprocessing_config_val,
    preprocessing_config_linear_decay_train,
    preprocessing_config_linear_decay_val
)

adversarial_training_denoising_subspace_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "attack_types" : ["pgd"]*4,
    "rec_operator_type" : "factor_zeros",
    "epochs"       : 50,
    "noise_level"  : 0.2,
    "zetas_start"  : 0.0,
    "zetas_end"    : 0.3,
    "zetas_steps"  : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "adversarial_training_denoising_subspace_config")
}

jittering_via_formula_denoising_subspace_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "hp_path" : None,
    "rec_operator_type" : "factor_zeros",
    "epochs" : 50,
    "noise_level" : 0.2,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jittering_via_formula_denoising_subspace_config")
}

jittering_via_hpsearch_denoising_subspace_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "hp_path" : os.path.join(hpsearch_denoising_jittering_subspace_eval["base_path_eval"], "mean_minimizers_last.csv"),
    "epochs" : 50,
    "rec_operator_type" : "factor_zeros",
    "noise_level" : 0.2,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jittering_via_hpsearch_denoising_subspace_config")
}

standard_denoising_subspace_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "epochs" : 50,
    "rec_operator_type" : "factor_zeros",
    "noise_level" : 0.2,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "standard_denoising_subspace_config")
}

adversarial_training_linear_decay_subspace_config = {
    "d_config_train" : preprocessing_config_linear_decay_train,
    "d_config_val" : preprocessing_config_linear_decay_val,
    "t_set_base" : training_base_config,
    "attack_types" : ["pgd"]*4,
    "rec_operator_type" : "diagonal",
    "epochs"       : 50,
    "noise_level"  : 0.2,
    "zetas_start"  : 0.0,
    "zetas_end"    : 0.15,
    "zetas_steps"  : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "adversarial_training_linear_decay_subspace_config")
}

jittering_via_hpsearch_linear_decay_subspace_config = {
    "d_config_train" : preprocessing_config_linear_decay_train,
    "d_config_val" : preprocessing_config_linear_decay_val,
    "t_set_base" : training_base_config,
    "hp_path" : os.path.join(hpsearch_linear_decay_jittering_subspace_eval["base_path_eval"], "mean_minimizers_last.csv"),
    "epochs" : 50,
    "rec_operator_type" : "diagonal",
    "noise_level" : 0.2,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.15,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jittering_via_hpsearch_linear_decay_subspace_config")
}

standard_linear_decay_subspace_config = {
    "d_config_train" : preprocessing_config_linear_decay_train,
    "d_config_val" : preprocessing_config_linear_decay_val,
    "t_set_base" : training_base_config,
    "epochs" : 50,
    "rec_operator_type" : "diagonal",
    "noise_level" : 0.2,
    "zetas_end" : 0.15,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "standard_linear_decay_subspace_config")
}

from generic.step import ExperimentStep
from experiments.subspace.steps.train.train_adversarial import adversarial_training
from experiments.subspace.steps.train.train_jittering import train_via_jittering
from experiments.subspace.steps.train.train_standard import train_standard

train_step_configs = {
    "adversarial_training_denoising_subspace" : ExperimentStep(
        step_func=adversarial_training,
        artifact_path=adversarial_training_denoising_subspace_config["base_artifact_path"],
        parameters=adversarial_training_denoising_subspace_config
    ),
    "jittering_via_formula_training_denoising_subspace" : ExperimentStep(
        step_func=train_via_jittering,
        artifact_path=jittering_via_formula_denoising_subspace_config["base_artifact_path"],
        parameters=jittering_via_formula_denoising_subspace_config
    ),
    "jittering_via_hpsearch_training_denoising_subspace" : ExperimentStep(
        step_func=train_via_jittering,
        artifact_path=jittering_via_hpsearch_denoising_subspace_config["base_artifact_path"],
        parameters=jittering_via_hpsearch_denoising_subspace_config
    ),
    "standard_training_denoising_subspace" : ExperimentStep(
        step_func=train_standard,
        artifact_path=standard_denoising_subspace_config["base_artifact_path"],
        parameters=standard_denoising_subspace_config
    ),
    "adversarial_training_linear_decay_subspace" : ExperimentStep(
        step_func=adversarial_training,
        artifact_path=adversarial_training_linear_decay_subspace_config["base_artifact_path"],
        parameters=adversarial_training_linear_decay_subspace_config
    ),
    "jittering_via_hpsearch_training_linear_decay_subspace" : ExperimentStep(
        step_func=train_via_jittering,
        artifact_path=jittering_via_hpsearch_linear_decay_subspace_config["base_artifact_path"],
        parameters=jittering_via_hpsearch_linear_decay_subspace_config
    ),
    "standard_training_linear_decay_subspace" : ExperimentStep(
        step_func=train_standard,
        artifact_path=standard_linear_decay_subspace_config["base_artifact_path"],
        parameters=standard_linear_decay_subspace_config
    )
}