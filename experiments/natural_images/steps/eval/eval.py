from random import random
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange

from torch.utils.data import Dataset, DataLoader
from experiments.natural_images.steps.eval.eval_single import eval_single_rundir

from generic.helper.device_helpers import get_last_gpu__or__cpu
from generic.helper.pandas_helper import add_settings_info_to_df
from generic.helper.tb_helpers import pltfig_to_tensor
#from reconstruction_models.UNetReconstruction import UNet
import generic.reconstruction_models.UNetReconstruction as unetv1
import generic.reconstruction_models.UNetReconstructionV0 as unetv0
from generic.reconstruction_models.loss_functions import get_loss_fn
from generic.robust_training.AttackerModel import AttackerModel
from generic.signal_models.NoiseReconstructionDataset import NoiseReconstructionDataset
from generic.signal_models.TensorListDataset import TensorListDataset
from experiments.natural_images.steps.train_and_test import test, train
from generic.transforms.AddGaussianNoise import AddGaussianNoise

import configparser
import os
import io
import PIL.Image
from datetime import datetime
from glob import glob
import pandas as pd

from generic.helper.mp_helper import distr_configs_over_gpus2
from generic.transforms.ApplyConvolution import ApplyConvolution, create_kernel_by_name

def _read_zetas_from_first_rundir(run_dir, eval=False):
    config_path = os.path.join(run_dir, "settings")
    config = configparser.ConfigParser()
    config.read(config_path)
    t_set = config["Training"]
    if eval and "zetas_eval_start" in t_set and float(t_set["zetas_start"]) == float(t_set["zetas_end"]):
        zetas = np.linspace(float(t_set["zetas_eval_start"]), float(t_set["zetas_eval_end"]), int(t_set["zetas_eval_steps"]))
    else:
        zetas = np.linspace(float(t_set["zetas_start"]), float(t_set["zetas_end"]), int(t_set["zetas_steps"]))
    return zetas

#def _calc_eps_for_alpha_xis(n, d, alphas, xis):
    #return np.sqrt(d) * ( (np.square(alphas) - np.square(xis)) / np.sqrt(np.power(alphas, 4) + n*np.square(xis)))

#def _calc_eps_for_lamb_xis(n, d, lambs, xis):
    #return np.sqrt(d) * n * lambs / np.sqrt(n**2 * lambs**2 + n*(1+2*lambs)*xis**2 + xis**4)

def evaluate(base_dir, fixed_noise_levels, nr_per_device = 1,
    df_save_dir = None, test_dataset_config = None, batch_size = 1,
    d_set_mapping = None, t_set_mapping = None,
    adv_its=5, zetas_by_noise_level = False, zetas_by_wd = False,
    adv_test = False, std_test = False,
    df_filename_adv = "adv.csv", df_filename_std = "std.csv", devices = ["cpu"]):

    base_dir_glob = os.path.join(base_dir, "*", "*")

    if adv_test:
        print(f"running adv_test")
        eval_rundirs(base_dir_glob, fixed_noise_levels, devices, nr_per_device, df_save_dir, df_filename_adv, test_dataset_config, batch_size, d_set_mapping, t_set_mapping, True, adv_its, zetas_by_noise_level, zetas_by_wd)

    if std_test:
        print(f"running std_test")
        eval_rundirs(base_dir_glob, fixed_noise_levels, devices, nr_per_device, df_save_dir, df_filename_std, test_dataset_config, batch_size, d_set_mapping, t_set_mapping, False, adv_its, zetas_by_noise_level, zetas_by_wd)

def eval_rundirs(base_dir, fixed_noise_levels, devices, nr_per_device = 1, df_save_dir = None, df_filename = None, test_dataset_config = None, batch_size = 1, d_set_mapping = None, t_set_mapping = None, adv_test=False, adv_its=5, zetas_by_noise_level = False, zetas_by_wd = False):
    run_dirs = glob(base_dir)

    if len(run_dirs) == 0:
        print(f"No run dirs found at: {base_dir}")
        return

    zetas = _read_zetas_from_first_rundir(run_dirs[0], eval=True) 

    tensor = torch.zeros( (len(run_dirs), len(fixed_noise_levels), len(zetas)) )
    tensor.share_memory_()

    adv_random_start = False
    adv_restarts = 0

    params = []
    for ind, run_dir in enumerate(run_dirs):
        params.append( (run_dir, test_dataset_config, tensor, fixed_noise_levels, adv_test, adv_its, adv_random_start, adv_restarts, zetas, None, None, True, batch_size, None) ) # None means latest epoch here and no fixed model path and new test dataset

    distr_configs_over_gpus2(devices, params, eval_single_rundir, nr_per_device)

    # loss_zeta_series_per_run: run, noise_level, zeta: loss --> zeta: loss
    data = tensor.detach().cpu().numpy() # shape: (len(runs), len(noise_levels), len(zeta))
    print(data.shape)
    r, n, z = data.shape
    out_arr = np.column_stack( ((np.repeat(np.arange(r), n)), np.tile(np.array(fixed_noise_levels), r), data.reshape(r*n, -1)) )
    column_names = ["run_dir", "sigma_te"] + list(map(lambda x: f"zeta={x:.2}", zetas))
    out_df = pd.DataFrame(out_arr, columns=column_names)

    # Replace run indices with actual paths
    for ind, run in enumerate(run_dirs):
        out_df.loc[out_df.run_dir == ind, "run_dir"] = run

    # add settings info to df (optional)
    if d_set_mapping is not None or t_set_mapping is not None:
        add_settings_info_to_df(out_df, "run_dir", d_set_mapping=d_set_mapping, t_set_mapping=t_set_mapping)

    if df_save_dir is not None and df_filename is not None:
        os.makedirs(df_save_dir, exist_ok=True)
        out_df.to_csv(os.path.join(df_save_dir, df_filename))
    
    return out_df