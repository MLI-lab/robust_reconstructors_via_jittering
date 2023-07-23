# %%
import torch.multiprocessing as mp
from experiments.subspace.steps.train.train_single import run_uc_subspace_robust_reconstruction
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

from experiments.subspace.configs.config_helper import set_operator_type

################################
#### Single run methods
################################
def run_training_via_jittering(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, sigma, test_sigma, zeta, rec_operator_type, epochs, zeta_end, zetas_steps, tensor, dir = params[index]
    #nr, sigma, test_sigma, zeta, d, n, epochs, zetas_end, zetas_steps, linear_forward, rec_operator_type, tensor = params[index]
    device = devices[pindex]
    t_set = t_set_base.copy()
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device
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

    set_operator_type(rec_operator_type, t_set)

    t_set["SGD_weight_decay"] = 0
    t_set["train_loss_fn_name"] = "mse"
    t_set["train_loss_fn_params"] = {}
    t_set["test_loss_fn_name"] = "mse"
    t_set["test_loss_fn_params"] = {}
    t_set["tb_subfolder"] = f"hpsearch_subspace_jittering_{sigma}nlt_{test_sigma}nl_{t_set['epochs']}e_{rec_operator_type}rot_color_adam2_{t_set['train_dataloader_batch_size']}bs/"
    print(t_set["tb_subfolder"])
    _, test_res = run_uc_subspace_robust_reconstruction(d_config_train, d_config_val, t_set, dir)
    tensor[nr] = np.min(np.array(test_res))

def run_hpsearch_train(
    d_config_train,
    d_config_val,
    t_set_base,
    devices,
    method_name,
    base_path_runs,
    base_path_eval,
    rec_operator_type = "factor_zeros",
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
        method = run_training_via_jittering
    else:
        raise Exception(f"Method: {method_name} not supported.")

    alphas = np.linspace(alpha_start, alpha_end, grid_size)

    print(f"Training of {grid_size} x {reps} models")
    for i in range(reps):
        print(f"Train for rep {i+1} of {reps}")

        tensor = torch.zeros((len(alphas)))
        tensor.share_memory_()
        params = list(zip(range(len(alphas)), [d_config_train]*len(alphas), [d_config_val]*len(alphas), [t_set_base]*len(alphas), list(alphas), [noise_level]*len(alphas), [0]*len(alphas), [rec_operator_type]*len(alphas), [epochs]*len(alphas), [zetas_end]*len(alphas), [zetas_steps]*len(alphas), [tensor]*len(alphas), [base_path_runs]*len(alphas)))

        distr_configs_over_gpus2(devices, params, method, nr_per_device=1)
        test_data = tensor.detach().cpu().numpy()
        print(f"Got test data: {test_data}")