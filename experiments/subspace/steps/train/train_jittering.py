import math
from collections import namedtuple
from itertools import repeat

import numpy as np
import torch.multiprocessing as mp
import pandas as pd

from experiments.subspace.configs.config_helper import set_operator_type

from generic.helper.mp_helper import distr_configs_over_gpus2
from experiments.subspace.steps.train.train_single import \
    run_uc_subspace_robust_reconstruction

def run_training_via_jittering(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, sigma, test_sigma, zeta, rec_operator_type, epochs, zetas_start, zeta_end, zetas_steps, label, base_artifact_path = params[index]
    device = devices[pindex]
    t_set = t_set_base.copy()
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device
    # device config
    t_set["info"] = f"red_dataset_mp_{device}_sigma_{sigma}"
    #t_set["expected_signal_norm_sq"] = sig_energy
    t_set["train_noise_level"] = sigma
    t_set["val_noise_level"] = sigma
    t_set["train_make_adv"] = False
    t_set["test_make_adv"] = False
    t_set["epochs"] = epochs
    t_set["zetas_start"] = zeta
    t_set["zetas_end"] = zeta
    t_set["zetas_steps"] = 1
    t_set["zetas_eval_start"] = zetas_start
    t_set["zetas_eval_end"] = zeta_end
    t_set["zetas_eval_steps"] = zetas_steps
    set_operator_type(rec_operator_type, t_set)
    t_set["SGD_weight_decay"] = 0
    t_set["train_loss_fn_name"] = "mse"
    t_set["train_loss_fn_params"] = {}
    t_set["test_loss_fn_name"] = "mse"
    t_set["test_loss_fn_params"] = {}
    t_set["tb_subfolder"] = f"uc_subspace_jittering_{label}_{test_sigma}nl_{t_set['epochs']}e_{rec_operator_type}rot_color_adam2_{t_set['train_dataloader_batch_size']}bs/"
    print(t_set["tb_subfolder"])
    run_uc_subspace_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path)

def calc_alphas__for__eps_xi_train(n, d, eps, xis):
    return np.sqrt( (d*xis**2 + np.sqrt(n)*eps*xis*np.sqrt(d-eps**2+d/n*xis**2)) / (n * d*np.ones_like(eps) - n*eps**2) )

def calc_alphas__for__eps_xi_train_add(n, d, eps, xis):
    return np.sqrt( (eps * xis / math.sqrt(n) * (np.ones_like(xis) + xis**2 / n)) / (np.sqrt(d - eps**2 + d / n * xis**2) - eps * xis / math.sqrt(n)))

def calc_jittering_l2_for__eps_xi_train(n, d, eps, xis):
    return np.square(calc_alphas__for__eps_xi_train_add(n, d, eps, xis))

def calc_jittering(n, d, sigma_c, sigma_z, eps):
    return np.sqrt(eps**2 * sigma_z**2 * d/n + sigma_z * sigma_c * eps * math.sqrt(d/n) * np.sqrt(sigma_c**2 - eps**2 + sigma_z**2 * d/n)) / np.sqrt(d * (sigma_c**2 - eps**2))

def get_jittering_values_via_config(hp_path, zetas, n = None, d = None, noise_level=None, sig_energy=None):
    if hp_path == None:
        if d == None:
            d = sig_energy
        print(f"No hp path provided, use formula from linear theory.")
        ones_np = np.ones_like(zetas)
        sigma_c = math.sqrt(sig_energy)
        sigma_z = noise_level * math.sqrt(n)
        eps = np.sqrt(zetas) * sigma_c
        alphas = calc_jittering(n, d, ones_np*sigma_c, ones_np*sigma_z, eps)
        alphas = np.sqrt(alphas**2 + noise_level**2)
        label = "formula"
    else:
        hps = pd.read_csv(hp_path).to_numpy()
        alphas = hps[:,1]
        label = "hp_opt"
    return alphas, label

def train_via_jittering(
    d_config_train, d_config_val, t_set_base,
    hp_path = None, rec_operator_type = "factor_zeros", epochs=100, sigc = None, noise_level=0.2, zetas_start = 0.0, zetas_end = 0.3, zetas_steps=11, devices = ["cpu"], base_artifact_path="runs"):

    N = zetas_steps
    zetas = np.linspace(0, zetas_end, N)

    sig_energy = d_config_train["d"]
    n, d = d_config_train["n"], d_config_train["d"]
    alphas, label = get_jittering_values_via_config(hp_path, zetas, n, d, noise_level, sig_energy)

    params = list(zip(range(len(alphas)), [d_config_train]*len(alphas), [d_config_val]*len(alphas), [t_set_base]*len(alphas), list(alphas), [noise_level]*len(alphas), list(zetas),  [rec_operator_type]*len(alphas), [epochs]*len(alphas), [zetas_start]*len(alphas), [zetas_end]*len(alphas), [zetas_steps]*len(alphas), [label]*len(alphas), [base_artifact_path]*len(alphas)))
    print(f"Alphas: {alphas}")
    distr_configs_over_gpus2(devices, params, run_training_via_jittering, nr_per_device=1)