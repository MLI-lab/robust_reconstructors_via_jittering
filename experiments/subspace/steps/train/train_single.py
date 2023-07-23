# %%
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange

from torch.utils.tensorboard import SummaryWriter
from generic.helper.device_helpers import get_last_gpu__or__cpu
from generic.helper.tb_helpers import pltfig_to_tensor
from generic.reconstruction_models.UNetReconstruction import UNet
from generic.reconstruction_models.loss_functions import get_loss_fn
from generic.robust_training.AttackerModel import AttackerModel
from generic.signal_models.NoiseReconstructionDataset import NoiseReconstructionDataset
from generic.signal_models.TensorListDataset import TensorListDataset

from experiments.subspace.steps.train_and_test import test, train
from generic.transforms.AddGaussianNoise import AddGaussianNoise
from generic.transforms.ApplyConvolution import ApplyConvolution, create_kernel_by_name

from experiments.subspace.steps.preprocessing import (
    dataset_via_config
)

from generic.reconstruction_models.SubspaceProjectionNetwork import NeuralNetwork

import configparser
import os
from datetime import datetime

def setup_logdir(t_set, base_artifact_path):

    trainInfo = f"train{'Adv' if t_set['train_make_adv'] else 'Std'}{t_set['train_noise_level']}nl_{t_set['train_dataloader_batch_size']}bs"
    testInfo = f"test{'Adv' if t_set['test_make_adv'] else 'Std'}{t_set['val_noise_level']}nl_{t_set['test_dataloader_batch_size']}bs"

    now = datetime.now()
    timestamp_now = f"{now.year}-{now.month}-{now.day}__{now.hour}-{now.minute}-{now.second}"
    log_dir = f"{base_artifact_path}/{t_set['tb_subfolder']}/{os.path.basename(__file__)}_{t_set['fixed_gpu_device']}_{trainInfo}_{testInfo}_{t_set['zetas_start']:.5f}-{t_set['zetas_end']:.5f}-{t_set['zetas_steps']}z_{t_set['epochs']}e_{timestamp_now}"
    return log_dir

def save_config(log_dir, d_config_train, d_config_val, t_set):
    # %% save config
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config = configparser.ConfigParser()
    config.add_section("Data")
    config["DataTrain"] = d_config_train
    config["DataVal"] = d_config_val
    config["Training"] = t_set
    with open(os.path.join(log_dir, "settings"), "w") as configfile:
        config.write(configfile)

def create_temp_file(log_dir):
    sr_path = os.path.join(log_dir, "started_run.txt")
    if os.path.exists(sr_path):
        print(f"sr_path already exists: {sr_path}")
    else:
        f = open(sr_path, "x")
        f.write("Is running..")
        f.close()

def get_device(t_set):
    if not bool(t_set["data_parallel"]):
        if bool(t_set["fixed_gpu"]):
            device = t_set["fixed_gpu_device"]
        else:
            device = "cpu"
    else:
        device = eval(t_set["data_parallel_list"])[0]
    return device

from experiments.subspace.steps.preprocessing import dataset_via_config

def run_uc_subspace_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path):
    log_dir = setup_logdir(t_set, base_artifact_path)

    print(f"Setup log_dir: {log_dir}")
    save_config(log_dir, d_config_train, d_config_val, t_set) 
    create_temp_file(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    device = get_device(t_set)

    # %%
    factors = []
    loss_train_series_per_epoch = []
    loss_test_series_per_epoch = []
    mses_test = []

    t_set["zetas"] = np.linspace(t_set["zetas_start"], t_set["zetas_end"], t_set["zetas_steps"])
    zetas = t_set["zetas"]

    dataset_train = dataset_via_config(d_config_train, device=device, noise_level=t_set["train_noise_level"])
    dataset_val = dataset_via_config(d_config_val, device=device, noise_level=t_set["val_noise_level"])

    n = d_config_train["n"]
    d = d_config_train["d"]

    zeta_index = 0
    for zeta in tqdm(zetas, position=0):

        eps = math.sqrt(zeta*d)

        # Init dataset:
        dataset_train_loader = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=t_set['train_dataloader_batch_size'], shuffle=True,
                                                    num_workers=t_set['num_workers'], pin_memory=t_set['train_pin_memory'])

        dataset_test_loader = torch.utils.data.DataLoader(dataset_val,
                                                    batch_size=t_set['test_dataloader_batch_size'], shuffle=False,
                                                    num_workers=t_set['num_workers'], pin_memory=t_set['test_pin_memory'])
        # Train network:

        if "train_loss_fn_name" in t_set:
            train_loss_fn_name = t_set["train_loss_fn_name"]
            train_loss_fn = get_loss_fn(train_loss_fn_name, t_set["train_loss_fn_params"])
            test_loss_fn_name = t_set["test_loss_fn_name"]
            test_loss_fn = get_loss_fn(test_loss_fn_name, t_set["test_loss_fn_params"])
        else:
            print("No train_loss_fn_name specified. Use MSELoss.")
            train_loss_fn = torch.nn.MSELoss()
            test_loss_fn = torch.nn.MSELoss()


        not_skip_linear_diagonal = t_set["not_skip_linear_diagonal"] if "not_skip_linear_diagonal" in t_set else False
        full_linear = t_set["full_linear"] if "full_linear" in t_set else False
        model_d = NeuralNetwork(d_config_train["n"], d_config_val["d"], device, t_set["skip_linear"], not_skip_linear_diagonal, full_linear).to(device)

        if t_set["data_parallel"]:
            model_d = torch.nn.DataParallel(model_d, device_ids=t_set["data_parallel_list"])

        adv_step_size_factor = float(t_set["adv_step_size_factor"]) if "adv_step_size_factor" in t_set else 2.5

        attackerModel = AttackerModel(model_d)
        attack_kwargs = {
            'constraint': t_set["adv_perturbation_type"], #'2',
            'eps': eps,
            'step_size': adv_step_size_factor*eps / t_set["adv_its"],
            'iterations': t_set["adv_its"],
            'random_start': bool(t_set["adv_random_start"]), # False,
            'random_restarts': bool(t_set["adv_random_restarts"]), #False,
            'use_best': bool(t_set["adv_use_best"])#, True
        }

        optimizer = optim.SGD(attackerModel.parameters(), lr=t_set['SGD_lr'], momentum=t_set['SGD_momentum'], weight_decay=t_set["SGD_weight_decay"])
        loss_train_values = []
        loss_test_values = []
    
        epochs = t_set["epochs"]

        for t in trange(epochs, desc=f"Training epochs for zeta={zeta}", unit="epoch", position=1):
            loss_train_epoch_t = train(dataset_train_loader, attackerModel, device, train_loss_fn, optimizer, t_set["train_make_adv"], **attack_kwargs)
            writer.add_scalars(f"Loss/train",  {
                f"zeta={zeta}" : loss_train_epoch_t}, t)
            loss_train_values.append(loss_train_epoch_t)
            loss_test_epoch_t = test(dataset_test_loader, attackerModel, device, test_loss_fn, t_set["test_make_adv"], **attack_kwargs)
            writer.add_scalars(f"Loss/test",  {
                f"zeta={zeta}" : loss_test_epoch_t}, t)
            loss_test_values.append(loss_test_epoch_t)

        # Extract parameters:
        params = model_d.parameters()
        factor = np.max(list(params)[0].cpu().detach().numpy())
        writer.add_scalar(f"Params/factor", factor, zeta_index)
        zeta_index += 1
        factors.append(factor)
        loss_train_series_per_epoch.append(loss_train_values)
        loss_test_series_per_epoch.append(loss_test_values)
        mses_test.append(loss_test_values[-1])
        writer.flush()

        # Save models
        if t_set['save_models']:
            model_dir = os.path.join(log_dir, t_set['save_models_subdir'])
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            path = os.path.join(model_dir, f"model_zeta_{zeta}.pt")
            torch.save({
                'model_state_dict': model_d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path)

    # %%
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    eps_squares = zetas*d
    zetas_n = np.divide(eps_squares, n*np.ones(len(zetas)))
    predicted_factors = np.divide(np.ones(len(zetas_n)), np.ones(len(zetas_n)) + zetas_n)

    ax[0].scatter(zetas, factors, s=8)
    ax[0].plot(zetas, factors, label="Robust Training Estimator (Exp.)")
    ax[0].scatter(zetas, predicted_factors, s=8)
    ax[0].plot(zetas, predicted_factors, label= "Average noise Estimator (Pred.)")
    ax[0].set_xlabel(r"$\zeta$")
    ax[0].set_ylabel(r"$\alpha$")
    ax[0].set_title(r"Scaling factors for $f_\alpha$ under robust Training.")
    ax[0].legend()

    predicted_mse = np.divide(d/n*np.array(zetas_n), np.ones(len(zetas_n)) + np.array(zetas_n))

    ax[1].scatter(zetas, mses_test, s=8)
    ax[1].plot(zetas, mses_test, label="Robust Training (Exp.")
    ax[1].scatter(zetas, predicted_mse, s=8)
    ax[1].plot(zetas, predicted_mse, label="Random Noise (Pred.)")
    ax[1].set_xlabel(r"$\zeta$")
    ax[1].set_ylabel("mse")
    ax[1].set_title("Coordinate-wise Error on Test Dataset.")
    ax[1].legend()

    fig.tight_layout()

    # %%
    writer.add_image("images/scaling_mse", pltfig_to_tensor(fig))

    # %%
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    epochs_indices = np.arange(epochs)
    for index, series_in_epoch in enumerate(loss_train_series_per_epoch):
        ax[0].plot(epochs_indices, series_in_epoch, label=r"$\zeta$=" + f"{zetas[index]}")

    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("mse")
    ax[0].set_title("Training")
    #ax[0].set_ylim([0,1])
    ax[0].grid()
    ax[0].legend()

    for index, series_in_epoch in enumerate(loss_test_series_per_epoch):
        ax[1].plot(epochs_indices, series_in_epoch, label=r"$\zeta$=" + f"{zetas[index]}")

    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("mse")
    ax[1].set_title("Test")
    #ax[1].set_ylim([0,1])
    ax[1].grid()
    ax[1].legend()

    # %%
    writer.add_image("images/training_metrics", pltfig_to_tensor(fig))

    # %% final cleanup and remove started file
    writer.close()
    os.remove(os.path.join(log_dir, "started_run.txt"))
    # %%
    return loss_train_series_per_epoch, loss_test_series_per_epoch