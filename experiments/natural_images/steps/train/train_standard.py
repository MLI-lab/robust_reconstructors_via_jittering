import math
from collections import namedtuple
from itertools import repeat

import numpy as np
import torch.multiprocessing as mp
import pandas as pd

from generic.helper.mp_helper import distr_configs_over_gpus2
from experiments.natural_images.steps.train.train_single import \
    run_imagenet_unet_robust_reconstruction

def run_standard_training(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, test_sigma, zeta, conv_type, sig_energy, epochs, zeta_end, zetas_steps, base_artifact_path = params[index]
    device = devices[pindex]
    t_set = t_set_base.copy()
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device
    t_set["info"] = f"red_dataset_mp_{device}_nr_{nr}_sigma_{test_sigma}"
    t_set["expected_signal_norm_sq"] = sig_energy
    t_set["train_noise_level"] = test_sigma
    t_set["val_noise_level"]  = test_sigma
    t_set["adv_its"] = 5
    t_set["train_make_adv"] = False
    t_set["test_make_adv"] = False
    t_set["epochs"] = epochs
    t_set["zetas_start"] = zeta
    t_set["zetas_end"] = zeta
    t_set["zetas_steps"] = 1
    t_set["zetas_eval_start"] = 0
    t_set["zetas_eval_end"] = zeta_end
    t_set["zetas_eval_steps"] = zetas_steps
    t_set["train_loss_fn_name"] = "mse"; t_set["train_loss_fn_params"] = {}
    t_set["test_loss_fn_name"] = "mse"; t_set["test_loss_fn_params"] = {}
    t_set["train_eval_conv_kernel"] = conv_type
    t_set["tb_subfolder"] = f"unet_standard_{test_sigma}nl_{t_set['epochs']}e_{conv_type}_{sig_energy}se_color_{t_set['train_optimizer_name']}o_{t_set['train_lr_scheduler_name']}lrs_{t_set['SGD_lr']}lr_{t_set['train_dataloader_batch_size']}/"
    print(t_set["tb_subfolder"])
    run_imagenet_unet_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path)

def train_standard(
    d_config_train, d_config_val, t_set_base,
    conv_type = "None", sig_energy=10278, epochs=50, noise_level=0.2, zetas_end=0.3, zetas_steps=11, devices=["cpu"], base_artifact_path = "runs"):
    N = zetas_steps
    zetas = np.linspace(0, zetas_end, N)
    params = list(zip(range(len(zetas)),[d_config_train]*len(zetas), [d_config_val]*len(zetas), [t_set_base]*len(zetas), [noise_level]*len(zetas), list(zetas), [conv_type]*len(zetas), [sig_energy]*len(zetas), [epochs]*len(zetas), [zetas_end]*len(zetas), [zetas_steps]*len(zetas), [base_artifact_path]*len(zetas)))
    distr_configs_over_gpus2(devices, params, run_standard_training, nr_per_device=1)