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

def run_adversarial_training(pindex, offset, devices, params):
    index = offset + pindex
    nr, d_config_train, d_config_val, t_set_base, noise_level, rec_operator_type, epochs, zetas_start, zeta_end, zetas_steps, attack_type, adv_its, base_artifact_path = params[index]
    device = devices[pindex]
    sigma = noise_level
    t_set = t_set_base.copy()
    t_set["fixed_gpu"] = True
    t_set["fixed_gpu_device"] = device
    t_set["info"] = f"mp_{device}_sigma_{sigma}"

    #t_set["expected_signal_norm_sq"] = sig_energy
    t_set["train_noise_level"] = sigma
    t_set["val_noise_level"] = sigma

    t_set["train_make_adv"] = True
    t_set["test_make_adv"] = False
    t_set["zeta_by_noise_level"] = False

    t_set["adv_random_start"] = False
    t_set["adv_step_size_factor"] = 2.5
    t_set["adv_random_mode"] = "uniform_in_sphere"
    t_set["adv_its"] = adv_its

    set_operator_type(rec_operator_type, t_set)

    #attack_type = "fast_fgsm_in_sphere"
    if attack_type == "fast_fgsm_in_sphere":
        print("use in sphere")
        t_set["adv_random_start"] = True
        t_set["adv_step_size_factor"] = 1.25
        t_set['adv_use_best'] = False
        t_set["adv_random_mode"] = "uniform_in_sphere"
    elif attack_type == "fast_fgsm_on_sphere":
        print("use on sphere")
        t_set["adv_random_start"] = True
        t_set["adv_step_size_factor"] = 1.25
        t_set['adv_use_best'] = False
        t_set["adv_random_mode"] = "uniform_on_sphere"
    elif attack_type == "fast_fgsm_zero_init":
        print("use zero init")
        t_set["adv_random_start"] = False
        t_set["adv_step_size_factor"] = 1.25
        t_set['adv_use_best'] = False
        t_set["adv_random_mode"] = "None"
    else:
        print("use standard settings for attack")

    t_set["epochs"] = epochs
    t_set["zetas_start"] = zetas_start
    t_set["zetas_end"] = zeta_end
    t_set["zetas_steps"] = zetas_steps
    t_set["train_loss_fn_name"] = "mse"
    t_set["train_loss_fn_params"] = {}
    t_set["test_loss_fn_name"] = "mse"
    t_set["test_loss_fn_params"] = {}
    #t_set["train_eval_conv_kernel"] = conv_type
    t_set["tb_subfolder"] = f"unet_adversarial_training_{sigma}nl_{t_set['epochs']}e_{rec_operator_type}rot_{attack_type}at_{adv_its}teits_color_{t_set['train_optimizer_name']}o_{t_set['train_lr_scheduler_name']}lrs_{t_set['train_dataloader_batch_size']}bs/"
    print(t_set['tb_subfolder'])
    run_uc_subspace_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path)

def adversarial_training(
    d_config_train, d_config_val, t_set_base, attack_types = ["pgd"]*4, rec_operator_type = "factor_zeros", epochs=100, noise_level = 0.2, zetas_start = 0.0, zetas_end = 0.3, zetas_steps=11, adv_its=5, devices=["cpu"], repetitions=1, base_artifact_path = "eval"):

    levels = [noise_level]*repetitions
    rec_operator_types = [rec_operator_type]*repetitions

    params = list(zip(range(len(levels)), [d_config_train]*len(levels), [d_config_val]*len(levels), [t_set_base]*len(levels), list(levels), list(rec_operator_types), [epochs]*len(levels), [zetas_start]*len(levels), [zetas_end]*len(levels), [zetas_steps]*len(levels), attack_types, [adv_its]*len(levels), [base_artifact_path]*len(levels)))

    distr_configs_over_gpus2(devices, params, run_adversarial_training, nr_per_device=1)