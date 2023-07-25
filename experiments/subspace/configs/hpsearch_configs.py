import os

from experiments.subspace.configs.base_config import (
    base_artifact_path__default
)

step_name = "hpsearch"

from experiments.subspace.configs.preprocess_configs import (
    preprocessing_config_train,
    preprocessing_config_val,
    preprocessing_config_linear_decay_train,
    preprocessing_config_linear_decay_val
)

t_set_base = {
    "epochs":20,
    "zetas_start" : 0,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "dataloader_split_train" : False,
    "train_dataset_percentage": 0.8,
    "train_dataloader_batch_size": 50,
    "test_dataloader_batch_size": 100,
    "num_workers": 0,
    "SGD_lr": 1e-1,
    "SGD_momentum": 0.9,
    "SGD_weight_decay" : 0,
    "adv_its": 20,
    "adv_perturbation_type" : "2",
    "adv_random_start" : True,
    "adv_random_restarts" : False,
    "adv_use_best" : False,
    "adv_random_mode" : "uniform_in_sphere",
    "adv_step_size_factor" : 2.5,
    "unet_classes" : 3,
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
    "fixed_gpu" : False,
    "fixed_gpu_device" : "cuda:1",
    "zeta_by_noise_level" : False,
    "zeta_by_wd" : False,
    "train_optimizer_name" : "Adam",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_args" : {"gamma" : 0.1, "step_size" : 25}
}


hpsearch_denoising_jittering_subspace_train = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "rec_operator_type" : "factor_zeros",
    "noise_level"       : 0.2,
    "epochs"            : 20,
    "grid_size"         : 24,
    "alpha_start"       : 0.2,
    "alpha_end"         : 0.6,
    "zetas_end"         : 0.3,
    "zetas_steps"       : 11,
    "reps"              : 1
}

hpsearch_denoising_jittering_subspace_eval = {
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "rec_operator_type" : "factor_zeros",
    "noise_level"       : 0.2,
    "epochs"            : 20,
    "grid_size"         : 24,
    "alpha_start"       : 0.2,
    "alpha_end"         : 0.6,
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

hpsearch_linear_decay_jittering_subspace_train = {
    "d_config_train" : preprocessing_config_linear_decay_train,
    "d_config_val" : preprocessing_config_linear_decay_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "rec_operator_type" : "diagonal",
    "noise_level"       : 0.2,
    "epochs"            : 20,
    "grid_size"         : 24,
    "alpha_start"       : 0.2,
    "alpha_end"         : 1.0,
    "zetas_end"         : 0.15,
    "zetas_steps"       : 11,
    "reps"              : 1
}

hpsearch_linear_decay_jittering_subspace_eval = {
    "d_config_val" : preprocessing_config_linear_decay_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "rec_operator_type" : "diagonal",
    "noise_level"       : 0.2,
    "epochs"            : 20,
    "grid_size"         : 24,
    "alpha_start"       : 0.2,
    "alpha_end"         : 1.0,
    "zetas_end"         : 0.15,
    "zetas_steps"       : 11,
    "reps"              : 1,
    "smoothing_N"        : 3,
    "adv_test"          : True,
    "adv_its"           : 10,
    "adv_random_start"  : False,
    "adv_restarts"      : 0,
    "eval_epoch_backcalc" : 0
}

def set_hpsearch_dirs(c, config_train_name, config_eval_name):
    c['base_path_runs'] = os.path.join(base_artifact_path__default, step_name, config_train_name)
    c['base_path_eval'] = os.path.join(base_artifact_path__default, step_name, config_eval_name)

set_hpsearch_dirs(hpsearch_denoising_jittering_subspace_train, "hpsearch_denoising_jittering_subspace_train", "hpsearch_denoising_jittering_subspace_eval")
set_hpsearch_dirs(hpsearch_denoising_jittering_subspace_eval, "hpsearch_denoising_jittering_subspace_train", "hpsearch_denoising_jittering_subspace_eval")
set_hpsearch_dirs(hpsearch_linear_decay_jittering_subspace_train, "hpsearch_linear_decay_jittering_subspace_train", "hpsearch_linear_decay_jittering_subspace_eval")
set_hpsearch_dirs(hpsearch_linear_decay_jittering_subspace_eval, "hpsearch_linear_decay_jittering_subspace_train", "hpsearch_linear_decay_jittering_subspace_eval")

from generic.step import ExperimentStep
from experiments.subspace.steps.hpsearch_train import run_hpsearch_train
from experiments.subspace.steps.hpsearch_eval import run_hpsearch_eval

step_configs = {
    "hpsearch_denoising_jittering_train_subspace" : ExperimentStep(
        step_func=run_hpsearch_train,
        artifact_path=hpsearch_denoising_jittering_subspace_train['base_path_runs'],
        parameters=hpsearch_denoising_jittering_subspace_train
    ),
    "hpsearch_denoising_jittering_eval_subspace" : ExperimentStep(
        step_func=run_hpsearch_eval,
        artifact_path=hpsearch_denoising_jittering_subspace_eval['base_path_eval'],
        parameters=hpsearch_denoising_jittering_subspace_eval
    ),
    "hpsearch_linear_decay_jittering_train_subspace" : ExperimentStep(
        step_func=run_hpsearch_train,
        artifact_path=hpsearch_linear_decay_jittering_subspace_train['base_path_runs'],
        parameters=hpsearch_linear_decay_jittering_subspace_train
    ),
    "hpsearch_linear_decay_jittering_eval_subspace" : ExperimentStep(
        step_func=run_hpsearch_eval,
        artifact_path=hpsearch_linear_decay_jittering_subspace_eval['base_path_eval'],
        parameters=hpsearch_linear_decay_jittering_subspace_eval
    )
}