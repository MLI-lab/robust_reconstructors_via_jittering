# %%
from random import random
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import math
from tqdm import tqdm
from tqdm import trange

import fastmri
import fastmri.models

from experiments.mri.steps.preprocessing import dataset_via_config

from generic.transforms.AddGaussianNoise import AddGaussianNoise

import configparser
import os

from generic.robust_training.AttackerModel import AttackerModel
from experiments.mri.steps.train_and_test import test, train
from generic.signal_models.MRITransformedDataset import MRITransformedDataset
from generic.transforms.mri_transform import data_transform

def eval_single_rundir(pindex, offset, devices, params):
    index = offset + pindex
    run_dir, test_dataset_config, result_tensor, fixed_noise_levels, adv_test, adv_its, adv_random_start, adv_restarts, zetas_eval, epoch, fixed_model_path, zeta_intersect, batch_size, test_dataset_per_device = params[index]
    device = devices[pindex]
    print(f"Evaluating rundir {run_dir} on device {device} (offset={offset}, pindex={pindex})")

    config_path = os.path.join(run_dir, "settings")
    config = configparser.ConfigParser()
    config.read(config_path)
    t_set = config["Training"]
    model_dir = os.path.join(run_dir, t_set["save_models_subdir"])
    zetas = np.linspace(float(t_set["zetas_start"]), float(t_set["zetas_end"]), int(t_set["zetas_steps"]))

    # in normal flow zetas is subset of zetas_eval, so this operation
    # should not do anything. It is used in convergence script for externally
    # constraining the zetas to evaluate on.
    if zeta_intersect:
        zetas = np.intersect1d(zetas, zetas_eval)
    else:
        zetas = zetas_eval
 
    #if "seed" in t_set:
        #seed = int(t_set["seed"])
        ## random sources: sampling of dataloader, network initialization
        #torch.manual_seed(seed)
        #random.seed(seed)
        #np.random.seed(seed)

    dataset_test_base = dataset_via_config(test_dataset_config)
    print(f"Test dataset has {len(dataset_test_base)} slices.")

    model_d = fastmri.models.Unet(in_chans=1, out_chans=1, chans=16, num_pool_layers=4, drop_prob=0.0).to(device)

    if epoch is None or int(t_set["epochs"]) <= epoch:
        epoch = None

    for noise_level_ind, noise_level in enumerate(fixed_noise_levels):
        for zeta_ind, zeta in enumerate(zetas):
            sig_energy = float(t_set["expected_signal_norm_sq"])
            eps = math.sqrt(zeta * sig_energy)

            print(f"Evalute at zeta={zeta} with eps={eps} and sig_energy={sig_energy}")
            print(f"Fixed model path is: {fixed_model_path}")
            filename = f"model_zeta_{zeta}.pt" if fixed_model_path is None else fixed_model_path
            if epoch is not None:
                model_path = os.path.join(model_dir, f"_{epoch}e", filename)
            else:
                model_path = os.path.join(model_dir, filename)

            model_dict = torch.load(model_path, map_location=device)
            model_state_dict = model_dict["model_state_dict"]
            model_d.load_state_dict(model_state_dict)
            #model_d = model_d.to(device)
            attackerModel = AttackerModel(model_d)
            random_start = adv_random_start
            random_restart = adv_restarts
            factor = 2.5
            use_best = False
            attack_kwargs = {
                'constraint': '2',
                'eps': eps,
                'step_size': factor*eps / float(adv_its),
                'iterations': int(adv_its),
                'random_start': random_start,
                'random_restarts': random_restart,
                'use_best': use_best,
                'random_mode' : "uniform_in_sphere",
                'data_transform' : data_transform
            }

            transform = AddGaussianNoise(mean=0, std=noise_level) if noise_level != 0.0 else None

            dataset_test = MRITransformedDataset(dataset=dataset_test_base, transform=transform, device=device)

            print(f"Noise level={noise_level}, adv_its={adv_its}, adv_random_start={adv_random_start}, adv_restarts={adv_restarts}, test_batch_size={batch_size}")

            dataset_test_loader = torch.utils.data.DataLoader(dataset_test,
                batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False)
                                        
            # Train network:
            #if "test_loss_fn_name" in t_set:
                #test_loss_fn_name = t_set["test_loss_fn_name"]
                #test_loss_fn = get_loss_fn(test_loss_fn_name, eval(t_set["test_loss_fn_params"]))
            #else:
                ##print("No train_loss_fn_name specified. Use MSELoss.")
                #test_loss_fn = get_loss_fn("mse", {})

            test_loss_fn = torch.nn.MSELoss()

            loss_test_epoch_t = test(dataset_test_loader, attackerModel, device, test_loss_fn, adv_test, **attack_kwargs)
            zeta_ind_true = list(zetas_eval).index(zeta)
            noise_ind_true = noise_level_ind
            result_tensor[index, noise_ind_true, zeta_ind_true] = loss_test_epoch_t