from experiments.natural_images.configs.base_config import (
    base_artifact_path__default
)
from experiments.natural_images.configs.hpsearch_configs import (
    hpsearch_denoising_jittering_rgb_eval,
    hpsearch_gaussian8x8_jittering_rgb_eval
)

import os
step_name = "train"

training_base_config = {
    "epochs":200,
    "zetas_start" : 0,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "dataloader_split_train" : False,
    "train_dataset_percentage": 0.8,
    "train_dataloader_batch_size": 50,
    "test_dataloader_batch_size": 50,
    "num_workers": 0,
    "SGD_lr": 1e-3,
    "SGD_momentum": 0.9,
    "SGD_weight_decay" : 0,
    "adv_its": 10,
    "adv_perturbation_type" : "2",
    "adv_random_start" : False,
    "adv_random_restarts" : False,
    "adv_use_best" : False,
    "adv_random_mode" : "uniform_in_sphere",
    "adv_step_size_factor" : 2.5,
    "enc_chs": (8, 16, 32, 64),
    "dec_chs": (64, 32, 16, 8),
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
    "tb_image_test_indices" : [50, 40, 80, 99],
    "tb_image_dpi" : 500,
    "tb_subfolder" : "",
    "save_models" : True,
    "save_models_subdir" : "models",
    "test_norm_filter_pseudo_len" : 351,
    "test_norm_filter_center" : 7972.48,
    "test_norm_filter_width" : 500,
    "train_norm_filter_pseudo_len" : 2894,
    "train_norm_filter_center" : 8157.58,
    "train_norm_filter_width" : 500,
    "fixed_gpu" : False,
    "fixed_gpu_device" : "cuda:1",
    "zeta_by_noise_level" : False,
    "zeta_by_wd" : False,
    "train_optimizer_name" : "Adam",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_args" : {"gamma" : 0.1, "step_size" : 25}
}

from experiments.natural_images.configs.preprocess_configs import (
    preprocessing_config_train,
    preprocessing_config_val
)

adversarial_training_denoising_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "conv_type"    : "None",
    "attack_types" : ["pgd"]*4,
    "sig_energy"   : 74500,
    "epochs"       : 200,
    "noise_level"  : 0.25,
    "zetas_start"  : 0.0,
    "zetas_end"    : 0.3,
    "zetas_steps"  : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "adversarial_training_denoising_config")
}

jittering_via_formula_denoising_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "hp_path" : None,
    "n" : 128*128*3,
    "d" : 32000,
    "conv_type" : "None",
    "sig_energy" : 74500,
    "epochs" : 200,
    "noise_level" : 0.25,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jittering_via_formula_denoising_config")
}

jittering_via_hpsearch_denoising_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "hp_path" : os.path.join(hpsearch_denoising_jittering_rgb_eval["base_path_eval"], "mean_minimizers_last.csv"),
    "n" : 128*128*3,
    "d" : 35000,
    "conv_type" : "None",
    "sig_energy" : 74500,
    "epochs" : 200,
    "noise_level" : 0.25,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jittering_via_hpsearch_denoising_config")
}

standard_denoising_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "conv_type" :  "None",
    "sig_energy" : 74500,
    "epochs" : 200,
    "noise_level" : 0.25,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "standard_denoising_config")
}

jacobian_denoising_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "conv_type" : "None",
    "sig_energy" : 74500,
    "hp_path" : "",
    "n" : 128*128*3,
    "d" : 128*128*3/2,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "noise_level" : 0.2,
    "epochs" : 200,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jacobian_denoising_config")
}

l2_denoising_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "d" : None,
    "conv_type" : "None",
    "sig_energy" : 74500,
    "hp_path" : None,
    "n" : 128*128*3,
    "d" : 128*128*3/2,
    "epochs" : 200,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "noise_level" : 0.25,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "l2_denoising_config")
}


adversarial_training_gaussian8x8_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "conv_type"    : "gaussian_8x8_s2_v1",
    "attack_types" : ["pgd"]*4,
    "sig_energy"   : 74500,
    "epochs"       : 200,
    "noise_level"  : 0.25,
    "zetas_start"  : 0.0,
    "zetas_end"    : 0.3,
    "zetas_steps"  : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "adversarial_training_gaussian8x8_config")
}

jittering_via_hpsearch_gaussian8x8_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "hp_path" : os.path.join(hpsearch_gaussian8x8_jittering_rgb_eval["base_path_eval"], "mean_minimizers_last.csv"),
    "n" : 128*128*3,
    "d" : 35000,
    "conv_type" : "gaussian_8x8_s2_v1",
    "sig_energy" : 74500,
    "epochs" : 200,
    "noise_level" : 0.25,
    "zetas_start" : 0.0,
    "zetas_end"   : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "jittering_via_hpsearch_gaussian8x8_config")
}

standard_gaussian8x8_config = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base" : training_base_config,
    "conv_type" :  "gaussian_8x8_s2_v1",
    "sig_energy" : 74500,
    "epochs" : 200,
    "noise_level" : 0.25,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "base_artifact_path" : os.path.join(base_artifact_path__default, step_name, "standard_gaussian8x8_config")
}

from generic.step import ExperimentStep
from experiments.natural_images.steps.train.train_adversarial import adversarial_training
from experiments.natural_images.steps.train.train_jittering import train_via_jittering
from experiments.natural_images.steps.train.train_standard import train_standard
from experiments.natural_images.steps.train.train_l2 import train_via_l2_regularization
from experiments.natural_images.steps.train.train_jacobian import train_via_jacobian_regularization

train_step_configs = {
    "adversarial_training_denoising_rgb" : ExperimentStep(
        step_func=adversarial_training,
        artifact_path=adversarial_training_denoising_config["base_artifact_path"],
        parameters=adversarial_training_denoising_config
    ),
    "jittering_via_formula_training_denoising_rgb" : ExperimentStep(
        step_func=train_via_jittering,
        artifact_path=jittering_via_formula_denoising_config["base_artifact_path"],
        parameters=jittering_via_formula_denoising_config
    ),
    "jittering_via_hpsearch_training_denoising_rgb" : ExperimentStep(
        step_func=train_via_jittering,
        artifact_path=jittering_via_hpsearch_denoising_config["base_artifact_path"],
        parameters=jittering_via_hpsearch_denoising_config
    ),
    "standard_training_denoising_rgb" : ExperimentStep(
        step_func=train_standard,
        artifact_path=standard_denoising_config["base_artifact_path"],
        parameters=standard_denoising_config
    ),
    "adversarial_training_gaussian8x8_rgb" : ExperimentStep(
        step_func=adversarial_training,
        artifact_path=adversarial_training_gaussian8x8_config["base_artifact_path"],
        parameters=adversarial_training_gaussian8x8_config
    ),
    "jittering_via_hpsearch_training_gaussian8x8_rgb" : ExperimentStep(
        step_func=train_via_jittering,
        artifact_path=jittering_via_hpsearch_gaussian8x8_config["base_artifact_path"],
        parameters=jittering_via_hpsearch_gaussian8x8_config
    ),
    "standard_training_gaussian8x8_rgb" : ExperimentStep(
        step_func=train_standard,
        artifact_path=standard_gaussian8x8_config["base_artifact_path"],
        parameters=standard_gaussian8x8_config
    )
}