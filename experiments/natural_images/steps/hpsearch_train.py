# %%
import torch.multiprocessing as mp
from experiments.natural_images.steps.train.train_single import run_imagenet_unet_robust_reconstruction
from generic.helper.mp_helper import distr_configs_over_gpus2
import math
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from glob import glob
import tikzplotlib as tplt

from generic.signal_models.TensorListDataset import TensorListDataset

#def average_over():
def calc_jittering(n, d, sigma_c, sigma_z, eps):
    return np.sqrt(eps**2 * sigma_z**2 * d/n + sigma_z * sigma_c * eps * math.sqrt(d/n) * np.sqrt(sigma_c**2 - eps**2 + sigma_z**2 * d/n)) / np.sqrt(d * (sigma_c**2 - eps**2))

################################
#### Single run methods
################################
def run_training_via_jittering_colorized(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, sigma, test_sigma, zeta, conv_type, sig_energy, epochs, zeta_end, zetas_steps, tensor, dir = params[index]
    #nr, sigma, test_sigma, zeta, d, n, epochs, zetas_end, zetas_steps, linear_forward, rec_operator_type, tensor = params[index]
    device = devices[pindex]
    t_set = t_set_base.copy()
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device

    t_set["expected_signal_norm_sq"] = sig_energy
    t_set["train_noise_level"] = sigma
    t_set["val_noise_level"] = sigma
    t_set["adv_its"] = 5
    t_set["train_make_adv"] = False
    t_set["test_make_adv"] = False
    t_set["epochs"] = epochs
    t_set["save_models_epochs"] =  True
    t_set["save_models"] = True
    t_set["save_models_epochs_mod"] = 1 # save at all epochs
    t_set["zetas_start"] = zeta
    t_set["zetas_end"] = zeta
    t_set["zetas_steps"] = 1
    t_set["zetas_eval_start"] = 0
    t_set["zetas_eval_end"] = zeta_end
    t_set["zetas_eval_steps"] = zetas_steps
    t_set["SGD_lr"] = 1e-2
    t_set["SGD_momentum"] = 0.9
    t_set["SGD_weight_decay"] = 0
    t_set["train_loss_fn_name"] = "mse"
    t_set["train_loss_fn_params"] = {}
    t_set["test_loss_fn_name"] = "mse"
    t_set["test_loss_fn_params"] = {}
    t_set["train_eval_conv_kernel"] = conv_type
    t_set["tb_subfolder"] = f"hpsearch_unet_jittering_{sigma}nlt_{test_sigma}nl_{t_set['epochs']}e_{conv_type}_{sig_energy}se_color_adam2_{t_set['train_dataloader_batch_size']}bs/"
    print(t_set["tb_subfolder"])
    _, test_res = run_imagenet_unet_robust_reconstruction(d_config_train, d_config_val, t_set, dir)
    tensor[nr] = np.min(np.array(test_res))

def run_training_via_l2_regularization_colorized(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, lam, test_sigma, zeta, conv_type, sig_energy, epochs, zeta_end, zetas_steps, tensor, dir = params[index]
    device = devices[pindex]
    t_set = t_set_base.copy()
    # device config
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device

    t_set["expected_signal_norm_sq"] = sig_energy
    t_set["train_noise_level"] = test_sigma
    t_set["val_noise_level"] = test_sigma

    t_set["adv_its"] = 5
    t_set["train_make_adv"] = False
    t_set["test_make_adv"] = False
    t_set["save_models_epochs"] =  False
    t_set["save_models"] = True
    t_set["save_models_epochs_mod"] = 1 # save at all epochs
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
    t_set["SGD_lr"] = 1e-4
    t_set["SGD_momentum"] = 0.9
    t_set["SGD_weight_decay"] = lam # scaling well acc. to hyperparameter seach
    t_set["train_eval_conv_kernel"] = conv_type
    t_set["tb_subfolder"] = f"hpsearch_unet_weight_decay_{lam}nlt_{test_sigma}nl_{epochs}e_{conv_type}_{sig_energy}se_color_adam2_{t_set_base['train_dataloader_batch_size']}bs/"
    _, test_res = run_imagenet_unet_robust_reconstruction(d_config_train, d_config_val, t_set, dir)
    tensor[nr] = np.min(np.array(test_res))

def train_models(d_set_base, t_set_base, run_single_method, alphas, noise_level, zeta, conv_type, sig_energy, epochs, base_dir, devices, zetas_end, zetas_steps):
    tensor = torch.zeros((len(alphas)))
    tensor.share_memory_()
    params = list(zip(range(len(alphas)), [d_set_base]*len(alphas), [t_set_base]*len(alphas), list(alphas), [noise_level]*len(alphas), [zeta]*len(alphas), [conv_type]*len(alphas), [sig_energy]*len(alphas), [epochs]*len(alphas), [zetas_end]*len(alphas), [zetas_steps]*len(alphas), [tensor]*len(alphas)))
    distr_configs_over_gpus2(devices, params, run_single_method, nr_per_device=1)
    return tensor.detach().cpu().numpy()

def run_hpsearch_train(
    d_config_train,
    d_config_val,
    t_set_base,
    devices,
    method_name,
    base_path_runs,
    base_path_eval,
    conv_type = "None",
    sig_energy = 74500,
    noise_level= 0.25,
    epochs = 40,
    grid_size = 41,
    alpha_start = 0.25,
    alpha_end = 0.9,
    zetas_end = 0.3,
    zetas_steps = 11,
    reps = 4
    ):

    if method_name == "jittering":
        method = run_training_via_jittering_colorized
    else:
        method = run_training_via_l2_regularization_colorized

    alphas = np.linspace(alpha_start, alpha_end, grid_size)

    print(f"Training of {grid_size} x {reps} models")
    for i in range(reps):
        print(f"Train for rep {i+1} of {reps}")

        tensor = torch.zeros((len(alphas)))
        tensor.share_memory_()
        params = list(zip(range(len(alphas)), [d_config_train]*len(alphas), [d_config_val]*len(alphas), [t_set_base]*len(alphas), list(alphas), [noise_level]*len(alphas), [0]*len(alphas), [conv_type]*len(alphas), [sig_energy]*len(alphas), [epochs]*len(alphas), [zetas_end]*len(alphas), [zetas_steps]*len(alphas), [tensor]*len(alphas), [base_path_runs]*len(alphas)))

        distr_configs_over_gpus2(devices, params, method, nr_per_device=1)
        test_data = tensor.detach().cpu().numpy()
        print(f"Got test data: {test_data}")