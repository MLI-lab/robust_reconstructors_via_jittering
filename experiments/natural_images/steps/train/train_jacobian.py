from experiments.natural_images.steps.train.train_jittering import get_jittering_values_via_config

import numpy as np

from generic.helper.mp_helper import distr_configs_over_gpus2
from experiments.natural_images.steps.train.train_single import \
    run_imagenet_unet_robust_reconstruction

def run_train_via_jacobian_regularization(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, lam, test_sigma, zeta, conv_type, sig_energy, epochs, zetas_end, zetas_steps, base_artifact_path, label = params[index]
    device = devices[pindex]
    t_set = t_set_base.copy()
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device
    t_set["info"] = f"red_dataset_mp_{device}_nr_{nr}_sigma_{test_sigma}_lam_{lam}"

    t_set["expected_signal_norm_sq"] = sig_energy
    t_set["train_noise_level"] = test_sigma
    t_set["val_noise_level"]  = test_sigma

    t_set["train_make_adv"] = False
    t_set["test_make_adv"] = False
    t_set["epochs"] = epochs
    t_set["zetas_start"] = zeta
    t_set["zetas_end"] = zeta
    t_set["zetas_steps"] = 1
    t_set["zetas_eval_start"] = 0
    t_set["zetas_eval_end"] = zetas_end
    t_set["zetas_eval_steps"] = zetas_steps
    t_set["SGD_weight_decay"] = 0
    t_set["train_eval_conv_kernel"] = conv_type
    t_set["train_loss_fn_name"] = "mse_error_reg"
    reg_param = lam
    t_set["train_loss_fn_params"] = {"base_param" : 1.0, "reg_param" : reg_param, "random_row" : True, "random_row_grad_output" : "normal"}
    t_set["ErrorReg_param"] = reg_param
    t_set["test_loss_fn_name"] = "mse"
    t_set["test_loss_fn_params"] = {}
    t_set["tb_subfolder"] = f"unet_jacobian_{test_sigma}nl_{label}lbl_{t_set['epochs']}e_{conv_type}_{sig_energy}se/"
    print(t_set["tb_subfolder"])
    run_imagenet_unet_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path)


def train_via_jacobian_regularization(
    d_config_train, d_config_val, t_set_base,
    hp_path = None, d=None, n = None, conv_type="None", sig_energy=10278, zetas_end=0.3, zetas_steps=11, noise_level=0.2, epochs=50, devices=["cpu"], base_artifact_path = "runs"):
    params = []

    alphas, label = get_jittering_values_via_config(hp_path, zetas, n, d, noise_level, sig_energy)

    N = zetas_steps
    zetas = np.linspace(0, zetas_end, N)

    lambdas = alphas**2

    params = list(zip(range(len(lambdas)), [d_config_train]*len(alphas), [d_config_val]*len(alphas), [t_set_base]*len(alphas), list(lambdas), [noise_level]*len(lambdas), list(zetas), [conv_type]*len(zetas), [sig_energy]*len(zetas), [epochs]*len(zetas), [zetas_end]*len(zetas), [zetas_steps]*len(zetas), [base_artifact_path]*len(zetas), [label]*len(zetas)))
    distr_configs_over_gpus2(devices, params, run_train_via_jacobian_regularization, nr_per_device=1)