import os

from experiments.mri.configs.base_config import (
    base_artifact_path__default
)

step_name = "hpsearch"

from experiments.mri.configs.preprocess_configs import (
    preprocessing_config_train,
    preprocessing_config_val
)

# %%
################################
#### Base configuration (multiple settings are deprecated and not used anymore)
################################
t_set_base = {
    "epochs":200,
    "zetas_start" : 0,
    "zetas_end" : 0.3,
    "zetas_steps" : 11,
    "dataloader_split_train" : False, # if false use default (train,test) separation of pytorch
    "train_dataset_percentage": 0.9,
    "train_dataloader_batch_size": 1, #4*500,
    "val_dataloader_batch_size": 1, #4*500,
    "num_workers": 0,
    "train_dataset_filtered" : True,
    "SGD_lr": 1e-2,
    "SGD_momentum": 0.9,
    "SGD_weight_decay" : 0,
    "adv_perturbation_type" : "2",
    "adv_random_start" : False,
    "adv_random_restarts" : False,
    "adv_use_best" : False,
    "adv_random_mode" : "uniform_in_sphere",
    "adv_step_size_factor" : 2.5,
    "train_make_adv" : False,
    "val_make_adv" : False,
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
    "train_optimizer_name" : "Adam",
    "train_lr_scheduler_name" : "None",
    "train_lr_scheduler_args" : {"gamma" : 0.1, "step_size" : 40},
    "save_tb_images_epochs" : True,
    "train_lr_scheduler_name" : "ReduceLROnPlateau",
    "train_lr_scheduler_args" : {"patience" : 5, "verbose" : 1},
    "train_subset_factor" : 1,
    "test_subset_factor" : 0.3,
    "train_test_split_factor" : 0.9,
    "train_dataloader_batch_size" : 1,
    "test_dataloader_batch_size" : 1,
    "save_tb_images_epochs" : True
}

hpsearch_jittering_mri_train = {
    "d_config_train" : preprocessing_config_train,
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "conv_type"         : "None",
    "sig_energy"        : 0.00077,
    "noise_level"       : 0.0,
    "epochs"            : 1,
    "grid_size"         : 24,
    "alpha_start"       : 0.00000,
    "alpha_end"         : 0.00002,
    "zetas_end"         : 0.03,
    "zetas_steps"       : 11,
    "reps"              : 1
}

hpsearch_jittering_mri_eval = {
    "d_config_val" : preprocessing_config_val,
    "t_set_base"        : t_set_base,
    "method_name"       : "jittering",
    "conv_type"         : "None",
    "sig_energy"        : 0.00077,
    "noise_level"       : 0.0,
    "epochs"            : 1,
    "grid_size"         : 24,
    "alpha_start"       : 0.00000,
    "alpha_end"         : 0.00002,
    "zetas_end"         : 0.03,
    "zetas_steps"       : 11,
    "reps"              : 1,
    "smoothing_N"        : 3,
    "adv_test"          : True,
    "adv_its"           : 5,
    "adv_random_start"  : False,
    "adv_restarts"      : 0,
    "eval_epoch_backcalc" : 0
}

def set_hpsearch_dirs(c):
    hpsearch_dir_name = f"hpsearch_{c['method_name']}_unet_{c['epochs']}e_{c['reps']}reps_{c['noise_level']}nl_{c['alpha_start']}as_{c['alpha_end']}ae_{c['zetas_end']}ze_{c['zetas_steps']}zs"
    c['hpsearch_dir_name'] = hpsearch_dir_name
    c['base_path_runs'] = os.path.join(base_artifact_path__default, step_name, "runs")#base_run_path, hpsearch_dir_name)
    c['base_path_eval'] = os.path.join(base_artifact_path__default, step_name, "eval")#base_eval_path, hpsearch_dir_name)
set_hpsearch_dirs(hpsearch_jittering_mri_train)
set_hpsearch_dirs(hpsearch_jittering_mri_eval)

from generic.step import ExperimentStep
from experiments.mri.steps.hpsearch_train import run_hpsearch_train
from experiments.mri.steps.hpsearch_eval import run_hpsearch_eval

step_configs = {
    "hpsearch_jittering_mri_train" : ExperimentStep(
        step_func=run_hpsearch_train,
        artifact_path=hpsearch_jittering_mri_train['base_path_runs'],
        parameters=hpsearch_jittering_mri_train
    ),
    "hpsearch_jittering_mri_eval" : ExperimentStep(
        step_func=run_hpsearch_eval,
        artifact_path=hpsearch_jittering_mri_eval['base_path_eval'],
        parameters=hpsearch_jittering_mri_eval
    ),
}