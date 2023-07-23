import math
from collections import namedtuple
from itertools import repeat

import numpy as np
import torch.multiprocessing as mp
import pandas as pd
from experiments.natural_images.steps.train.train_jittering import get_jittering_values_via_config

from generic.helper.mp_helper import distr_configs_over_gpus2
from experiments.natural_images.steps.train.train_single import \
    run_imagenet_unet_robust_reconstruction

def run_training_via_l2_regularization(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, t_set_base, lam, test_sigma, zeta, conv_type, sig_energy, epochs, zeta_end, zetas_steps, base_artifact_path = params[index]
    device = devices[pindex]
    t_set = t_set_base.copy()
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device
    t_set["expected_signal_norm_sq"] = sig_energy
    t_set["train_noise_level"] = test_sigma
    t_set["val_noise_level"]  = test_sigma
    t_set["info"] = f"red_dataset_mp_{device}_sigma_{test_sigma}_lam_{lam}"
    t_set["train_make_adv"] = False
    t_set["test_make_adv"] = False
    t_set["epochs"] = epochs
    t_set["zetas_start"] = zeta
    t_set["zetas_end"] = zeta
    t_set["zetas_steps"] = 1
    t_set["zetas_eval_start"] = 0
    t_set["zetas_eval_end"] = zeta_end
    t_set["zetas_eval_steps"] = zetas_steps
    t_set["train_loss_fn_name"] = "mse"
    t_set["train_loss_fn_params"] = {}
    t_set["test_loss_fn_name"] = "mse"
    t_set["test_loss_fn_params"] = {}
    t_set["SGD_weight_decay"] = 2*lam*1400.0 / 6400
    t_set["train_eval_conv_kernel"] = conv_type
    t_set["tb_subfolder"] = f"unet_weight_decay_{test_sigma}nl_{t_set['epochs']}e_{conv_type}_{sig_energy}se_colorized_adam2/"
    print(t_set["tb_subfolder"])
    run_imagenet_unet_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path)

def calc_alphas__for__eps_xi_train(n, d, eps, xis):
    return np.sqrt( (d*xis**2 + np.sqrt(n)*eps*xis*np.sqrt(d-eps**2+d/n*xis**2)) / (n * d*np.ones_like(eps) - n*eps**2) )

def calc_alphas__for__eps_xi_train_add(n, d, eps, xis):
    return np.sqrt( (eps * xis / math.sqrt(n) * (np.ones_like(xis) + xis**2 / n)) / (np.sqrt(d - eps**2 + d / n * xis**2) - eps * xis / math.sqrt(n)))

def calc_jittering_l2_for__eps_xi_train(n, d, eps, xis):
    return np.square(calc_alphas__for__eps_xi_train_add(n, d, eps, xis))

def calc_jittering(n, d, sigma_c, sigma_z, eps):
    return np.sqrt(eps**2 * sigma_z**2 * d/n + sigma_z * sigma_c * eps * math.sqrt(d/n) * np.sqrt(sigma_c**2 - eps**2 + sigma_z**2 * d/n)) / np.sqrt(d * (sigma_c**2 - eps**2))

def train_via_l2_regularization(
    d_config_train, d_config_val, t_set_base,
    hp_path = None, d=None, n = None, conv_type="None", sig_energy=10278, epochs=50, zetas_end=0.3, zetas_steps=11, noise_level=0.2, devices=["cpu"], base_artifact_path = "runs"):
    params = []

    alphas, label = get_jittering_values_via_config(hp_path, zetas, n, d, noise_level, sig_energy)

    #n = d_set_base["w"]*d_set_base["h"];
    #if d is None:
        #d = d_set_base["expected_signal_norm_sq_gray"]
    N = zetas_steps
    zetas = np.linspace(0, zetas_end, N)

    #lambdas = calc_jittering_l2_for__eps_xi_train(n, d, np.sqrt(zetas*d), noise_level*np.ones_like(zetas)*math.sqrt(n))

    lambdas = alphas**2

    params = list(zip(range(len(lambdas)), [d_config_train]*len(alphas), [d_config_val]*len(alphas), [t_set_base]*len(alphas), list(lambdas), [noise_level]*len(lambdas), list(zetas), [conv_type]*len(zetas), [sig_energy]*len(zetas), [epochs]*len(zetas), [zetas_end]*len(zetas), [zetas_steps]*len(zetas), [base_artifact_path]*len(zetas)))
    distr_configs_over_gpus2(devices, params, run_training_via_l2_regularization, nr_per_device=1)