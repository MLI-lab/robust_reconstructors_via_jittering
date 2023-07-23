import os

from experiments.natural_images.configs.base_config import (
    base_artifact_path__default
)

step_name = "hpsearch"

from experiments.natural_images.configs.preprocess_configs import (
    preprocessing_config_train,
    preprocessing_config_val
)

t_set_base = {
    "epochs":50,
    "zetas_start" : 0,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "dataloader_split_train" : False, # if false use default (train,test) separation of pytorch
    "train_dataset_percentage": 0.8,
    "train_dataloader_batch_size": 50,
    "test_dataloader_batch_size": 100,
    "num_workers": 0,
    "SGD_lr": 1e-3,
    "SGD_momentum": 0.9,
    "SGD_weight_decay" : 0,
    "adv_its": 10,
    "adv_perturbation_type" : "2",
    "adv_random_start" : True,
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
    "test_make_adv" : False,
    "train_pin_memory" : False,
    "test_pin_memory" : False,
    "data_parallel" : False,
    "data_parallel_list" : ["cpu"],
    "tb_image_test_indices" : [20, 40, 80, 99],
    "tb_image_dpi" : 100,
    "tb_subfolder" : "/",
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
    #"train_lr_scheduler_name" : "ExponentialLR",
    #"train_lr_scheduler_args" : {"gamma" : 0.95}
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_args" : {"gamma" : 0.1, "step_size" : 25}
}


hpsearch_denoising_jittering_rgb_train = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "conv_type"         : "None",
    "sig_energy"        : 74500,
    "noise_level"       : 0.25,
    "epochs"            : 50,
    "grid_size"         : 24,
    "alpha_start"       : 0.25,
    "alpha_end"         : 0.9,
    "zetas_end"         : 0.3,
    "zetas_steps"       : 11,
    "reps"              : 1
}

hpsearch_denoising_jittering_rgb_eval = {
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "conv_type"         : "None",
    "sig_energy"        : 74500,
    "noise_level"       : 0.25,
    "epochs"            : 50,
    "grid_size"         : 24,
    "alpha_start"       : 0.25,
    "alpha_end"         : 0.9,
    "zetas_end"         : 0.3,
    "zetas_steps"       : 11,
    "reps"              : 1,
    "smoothing_N"        : 3,
    "adv_test"          : True,
    "adv_its"           : 10,
    "adv_random_start"  : False,
    "adv_restarts"      : 0,
    "eval_epoch_backcalc" : 0
}

hpsearch_gaussian8x8_jittering_rgb_train = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "conv_type"         : "gaussian_8x8_s2_v1",
    "sig_energy"        : 74500,
    "noise_level"       : 0.25,
    "epochs"            : 50,
    "grid_size"         : 24,
    "alpha_start"       : 0.25,
    "alpha_end"         : 4.0,
    "zetas_end"         : 0.3,
    "zetas_steps"       : 11,
    "reps"              : 1
}

hpsearch_gaussian8x8_jittering_rgb_eval = {
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "conv_type"         : "gaussian_8x8_s2_v1",
    "sig_energy"        : 74500,
    "noise_level"       : 0.25,
    "epochs"            : 50,
    "grid_size"         : 24,
    "alpha_start"       : 0.25,
    "alpha_end"         : 4.0,
    "zetas_end"         : 0.3,
    "zetas_steps"       : 11,
    "reps"              : 1,
    "smoothing_N"        : 3,
    "adv_test"          : True,
    "adv_its"           : 5,
    "adv_random_start"  : False,
    "adv_restarts"      : 0,
    "eval_epoch_backcalc" : 0
}

def set_hpsearch_dirs(c, config_train_name, config_eval_name):
    c['base_path_runs'] = os.path.join(base_artifact_path__default, step_name, config_train_name)#base_run_path, hpsearch_dir_name)
    c['base_path_eval'] = os.path.join(base_artifact_path__default, step_name, config_eval_name)#base_eval_path, hpsearch_dir_name)

set_hpsearch_dirs(hpsearch_denoising_jittering_rgb_train, "hpsearch_denoising_jittering_rgb_train", "hpsearch_denoising_jittering_rgb_eval")
set_hpsearch_dirs(hpsearch_denoising_jittering_rgb_eval, "hpsearch_denoising_jittering_rgb_train", "hpsearch_denoising_jittering_rgb_eval")

set_hpsearch_dirs(hpsearch_gaussian8x8_jittering_rgb_train, "hpsearch_gaussian8x8_jittering_rgb_train", "hpsearch_gaussian8x8_jittering_rgb_eval")
set_hpsearch_dirs(hpsearch_gaussian8x8_jittering_rgb_eval, "hpsearch_gaussian8x8_jittering_rgb_train", "hpsearch_gaussian8x8_jittering_rgb_eval")

from generic.step import ExperimentStep
from experiments.natural_images.steps.hpsearch_train import run_hpsearch_train
from experiments.natural_images.steps.hpsearch_eval import run_hpsearch_eval

step_configs = {
    "hpsearch_denoising_jittering_train_rgb" : ExperimentStep(
        step_func=run_hpsearch_train,
        artifact_path=hpsearch_denoising_jittering_rgb_train['base_path_runs'],
        parameters=hpsearch_denoising_jittering_rgb_train
    ),
    "hpsearch_denoising_jittering_eval_rgb" : ExperimentStep(
        step_func=run_hpsearch_eval,
        artifact_path=hpsearch_denoising_jittering_rgb_eval['base_path_eval'],
        parameters=hpsearch_denoising_jittering_rgb_eval
    ),
    "hpsearch_gaussian8x8_jittering_train_rgb" : ExperimentStep(
        step_func=run_hpsearch_train,
        artifact_path=hpsearch_gaussian8x8_jittering_rgb_train['base_path_runs'],
        parameters=hpsearch_gaussian8x8_jittering_rgb_train
    ),
    "hpsearch_gaussian8x8_jittering_eval_rgb" : ExperimentStep(
        step_func=run_hpsearch_eval,
        artifact_path=hpsearch_gaussian8x8_jittering_rgb_eval['base_path_eval'],
        parameters=hpsearch_gaussian8x8_jittering_rgb_eval
    ),
}