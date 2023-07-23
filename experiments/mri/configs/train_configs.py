from experiments.mri.configs.base_config import (
    base_artifact_path__default
)

import os
step_name = "train"

from experiments.mri.configs.hpsearch_configs import hpsearch_jittering_mri_eval

training_base_config = {
    "epochs":200,
    "zetas_start" : 0,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "train_dataloader_batch_size": 1,
    "val_dataloader_batch_size": 1,
    "num_workers": 0,
    "adv_its": 1,
    "adv_perturbation_type" : "2",
    "adv_random_start" : False,
    "adv_random_restarts" : False,
    "adv_use_best" : False,
    "adv_random_mode" : "uniform_in_sphere",
    "adv_step_size_factor" : 2.5,
    "unet_classes" : 3,
    "train_make_adv" : True,
    "val_make_adv" : True,
    "train_pin_memory" : False,
    "val_pin_memory" : False,
    "data_parallel" : False,
    "data_parallel_list" : ["cpu"],
    "tb_image_val_indices" : [10, 40, 80, 100],
    "tb_image_dpi" : 100,
    "tb_subfolder" : "/",
    "save_models" : True,
    "save_models_subdir" : "models",
    "fixed_gpu" : False,
    "fixed_gpu_device" : "cuda:1",
    "zeta_by_noise_level" : False,
    "zeta_by_wd" : False,
    "SGD_lr": 1e-2,
    "SGD_momentum": 0.9,
    "SGD_weight_decay" : 0,
    "train_optimizer_name" : "Adam",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_name" : "ReduceLROnPlateau",
    "train_lr_scheduler_args" : {"patience" : 5, "verbose" : 1},
    "save_tb_images_epochs" : True
}

from experiments.mri.configs.preprocess_configs import (
    preprocessing_config_train,
    preprocessing_config_val
)

adversarial_training_mri_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "attack_types" : ["pgd"]*4,
    "sig_energy"   : 0.00077,
    "adv_its"      : 10,
    "epochs"       : 200,
    "noise_level"  : 0,
    "zetas_start"  : 0.0,
    "zetas_end"    : 0.03,
    "zetas_steps"  : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "adversarial_training_mri_config")
}


jittering_via_hpsearch_training_mri_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "hp_path" : os.path.join(hpsearch_jittering_mri_eval["base_path_eval"], "mean_minimizers_last.csv"),
    "sig_energy" : 0.00077,
    "epochs" : 200,
    "noise_level" : 0.0,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.03,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jittering_via_hpsearch_training_mri_config")
}

standard_training_mri_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "conv_type" :  "None",
    "sig_energy" : 0.00077,
    "epochs" : 200,
    "noise_level" : 0.0,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.03,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "standard_training_mri_config")
}

from generic.step import ExperimentStep
from experiments.mri.steps.train.train_adversarial import adversarial_training
from experiments.mri.steps.train.train_jittering import train_via_jittering
from experiments.mri.steps.train.train_standard import train_standard

train_step_configs = {
    "adversarial_training_mri" : ExperimentStep(
        step_func=adversarial_training,
        artifact_path=adversarial_training_mri_config["base_artifact_path"],
        parameters=adversarial_training_mri_config
    ),
    "jittering_via_hpsearch_training_mri" : ExperimentStep(
        step_func=train_via_jittering,
        artifact_path=jittering_via_hpsearch_training_mri_config["base_artifact_path"],
        parameters=jittering_via_hpsearch_training_mri_config
    ),
    "standard_training_mri" : ExperimentStep(
        step_func=train_standard,
        artifact_path=standard_training_mri_config["base_artifact_path"],
        parameters=standard_training_mri_config
    ),
}