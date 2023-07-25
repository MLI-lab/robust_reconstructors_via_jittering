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
from experiments.mri.steps.train_and_test import test, train
from generic.signal_models.MRITransformedDataset import MRITransformedDataset
from generic.signal_models.TensorListDataset import TensorListDataset
import random

from generic.transforms.mri_transform import data_transform

from generic.transforms.AddGaussianNoise import AddGaussianNoise

import fastmri
import fastmri.models

from experiments.mri.steps.preprocessing import (
    dataset_via_config
)

import configparser
import os
from datetime import datetime

def setup_logdir(t_set, base_artifact_path):

    trainInfo = f"train{'Adv' if t_set['train_make_adv'] else 'Std'}{t_set['train_noise_level']}nl_{t_set['train_dataloader_batch_size']}bs"
    valInfo = f"val{'Adv' if t_set['val_make_adv'] else 'Std'}{t_set['val_noise_level']}nl_{t_set['val_dataloader_batch_size']}bs"

    #n = d_set["n"]
    expected_signal_norm_sq = t_set["expected_signal_norm_sq"]
    now = datetime.now()
    timestamp_now = f"{now.year}-{now.month}-{now.day}__{now.hour}-{now.minute}-{now.second}"
    log_dir = f"{base_artifact_path}/{t_set['tb_subfolder']}/{os.path.basename(__file__)}_{t_set['fixed_gpu_device']}_{trainInfo}_{valInfo}_{expected_signal_norm_sq}esns_{t_set['zetas_start']:.5f}-{t_set['zetas_end']:.5f}-{t_set['zetas_steps']}z_{t_set['epochs']}e_{timestamp_now}"
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

def run_mri_unet_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path):

    #add_aux_entries_to_dicts(d_set, t_set)
    log_dir = setup_logdir(t_set, base_artifact_path)

    print(f"Setup log_dir: {log_dir}")
    save_config(log_dir, d_config_train, d_config_val, t_set) 
    create_temp_file(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # %% load device
    device = get_device(t_set)

    loss_train_series_per_epoch = []
    loss_val_std_series_per_epoch = []
    loss_val_adv_series_per_epoch = []
    mses_val_std = []
    mses_val_adv = []
    mses_train = []

    # seeds
    if "seed" in t_set:
        seed = int(t_set["seed"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    t_set["zetas"] = np.linspace(t_set["zetas_start"], t_set["zetas_end"], t_set["zetas_steps"])
    zetas = t_set["zetas"]

    dataset_train_base = dataset_via_config(d_config_train)
    dataset_val_base   = dataset_via_config(d_config_val)
    print(f"Using {len(dataset_train_base)} slices for training and {len(dataset_val_base)} for evaluation.")

    noise_level_train = t_set["train_noise_level"]
    noise_level_val = t_set["val_noise_level"]
    dataset_train = MRITransformedDataset(dataset=dataset_train_base, transform=AddGaussianNoise(mean=0, std=noise_level_train), device=device)
    dataset_val = MRITransformedDataset(dataset=dataset_val_base, transform=AddGaussianNoise(mean=0, std=noise_level_val), device=device)

    dataset_train_loader = torch.utils.data.DataLoader(dataset_train,
        batch_size=t_set['train_dataloader_batch_size'], shuffle=True,
        num_workers=t_set['num_workers'], pin_memory=t_set['train_pin_memory'])
                                        
    dataset_val_loader = torch.utils.data.DataLoader(dataset_val,
        batch_size=t_set['val_dataloader_batch_size'], shuffle=False,
        num_workers=t_set['num_workers'], pin_memory=t_set['val_pin_memory'])

    zeta_index = 0
    for zeta in tqdm(zetas, position=0):
        sig_energy = float(t_set["expected_signal_norm_sq"])
        eps = math.sqrt(zeta * sig_energy)

        # Train network:
        #if "train_loss_fn_name" in t_set:
            #train_loss_fn_name = t_set["train_loss_fn_name"]
            #train_loss_fn = get_loss_fn(train_loss_fn_name, t_set["train_loss_fn_params"])
            #val_loss_fn_name = t_set["val_loss_fn_name"]
            #val_loss_fn = get_loss_fn(val_loss_fn_name, t_set["val_loss_fn_params"])
        #else:
            #print("No train_loss_fn_name specified. Use MSELoss.")
        train_loss_fn = torch.nn.MSELoss()
        val_loss_fn = torch.nn.MSELoss()

        model_d = fastmri.models.Unet(in_chans=1, out_chans=1, chans=16, num_pool_layers=4, drop_prob=0.0).to(device)
    
        attackerModel = AttackerModel(model_d)
        if "adv_step_size_factor" in t_set:
            adv_step_size_factor = float(t_set["adv_step_size_factor"])
        else:
            adv_step_size_factor = 1

        adv_use_best = bool(t_set["adv_use_best"])
        attack_kwargs = {
            'constraint': t_set["adv_perturbation_type"],
            'eps': eps,
            'step_size': adv_step_size_factor * (eps / t_set["adv_its"]),
            'iterations': t_set["adv_its"],
            'random_start': bool(t_set["adv_random_start"]),
            'random_restarts': bool(t_set["adv_random_restarts"]),
            'use_best': adv_use_best,
            'random_mode' : t_set["adv_random_mode"],
            'data_transform' : data_transform
        }
        print(f"Train with step size = {adv_step_size_factor} and use_best = {adv_use_best}, wd = {t_set['SGD_weight_decay']}")

        optimizer = None
        if "train_optimizer_name" in t_set:
            optimizer_key = t_set["train_optimizer_name"]
            if optimizer_key == "SGD":
                optimizer = optim.SGD(attackerModel.parameters(), lr=t_set['SGD_lr'], momentum=t_set['SGD_momentum'], weight_decay=t_set["SGD_weight_decay"])
            else:
                optimizer = optim.Adam(attackerModel.parameters(), lr=t_set['SGD_lr'], weight_decay=t_set["SGD_weight_decay"])
        else: 
            print("No optimizer specified, use default.")
            optimizer = optim.Adam(attackerModel.parameters(), lr=t_set['SGD_lr'], weight_decay=t_set["SGD_weight_decay"])

        scheduler = None
        if "train_lr_scheduler" in t_set:
            scheduler_key = t_set["train_lr_scheduler_name"]
            scheduler_args = t_set["train_lr_scheduler_args"]
            if scheduler_key == "ExponentialLR":
                scheduler = optim.ExponentialLR(optimizer, gamma=float(scheduler_args["gamma"]))
            elif scheduler_key == "MultiStepLR":
                scheduler = optim.MultiStepLR(optimizer, milestones=scheduler_args["milestones"], gamma=scheduler_args["gamma"])
            elif scheduler_key == "StepLR":
                scheduler = optim.StepLR(optimizer, scheduler_args["step_size"], gamma=scheduler_args["gamma"], verbose=1)
            elif scheduler_key == "ReduceLROnPlateau":
                scheduler = optim.StepLR(optimizer, **scheduler_args)
            elif scheduler_key == "None":
                pass
            else:
                print(f"Unsupported scheduler: {scheduler_key}")

        loss_train_values = []
        loss_val_std_values = []
        loss_val_adv_values = []
    
        epochs = t_set["epochs"]

        for t in trange(epochs, desc=f"Training epochs for zeta={zeta}", unit="epoch", position=1):

            loss_train_epoch_t = train(dataset_train_loader, attackerModel, device, train_loss_fn, optimizer, t_set["train_make_adv"], **attack_kwargs)
            if scheduler != None:
                scheduler.step(loss_train_epoch_t)
                #scheduler.step()

            writer.add_scalars(f"lr",  {
                f"zeta={zeta}" : optimizer.state_dict()["param_groups"][0]["lr"]}, t)

            writer.add_scalars(f"Loss/train",  {
                f"zeta={zeta}" : loss_train_epoch_t}, t)
            loss_train_values.append(loss_train_epoch_t)

            loss_val_std_epoch_t = test(dataset_val_loader, attackerModel, device, val_loss_fn, False, **attack_kwargs)
            writer.add_scalars(f"Loss/val_std",  {
                f"zeta={zeta}" : loss_val_std_epoch_t}, t)
            loss_val_std_values.append(loss_val_std_epoch_t)

            loss_val_adv_epoch_t = test(dataset_val_loader, attackerModel, device, val_loss_fn, True, **attack_kwargs)
            writer.add_scalars(f"Loss/val_adv",  {
                f"zeta={zeta}" : loss_val_adv_epoch_t}, t)
            loss_val_adv_values.append(loss_val_adv_epoch_t)

            if "save_models_epochs" in t_set and t_set["save_models_epochs"]:
                if t % int(t_set["save_models_epochs_mod"]) == 0:
                    model_dir = os.path.join(log_dir, t_set['save_models_subdir'], f"_{t}e")
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    path = os.path.join(model_dir, f"model_zeta_{zeta}.pt")
                    torch.save({
                        'model_state_dict': model_d.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)

            if "save_tb_images_epochs" in t_set and t_set["save_tb_images_epochs"]:
                fig = create_example_fig(t_set["tb_image_val_indices"], zeta, dataset_val, model_d)
                writer.add_image(f"images/reconstruction_{zeta}zeta", pltfig_to_tensor(fig, dpi=t_set["tb_image_dpi"]), t)
                plt.close(fig)
                
        # Save values:
        loss_train_series_per_epoch.append(loss_train_values)
        mses_train.append(loss_train_values[-1])
        loss_val_std_series_per_epoch.append(loss_val_std_values)
        mses_val_std.append(loss_val_std_values[-1])
        loss_val_adv_series_per_epoch.append(loss_val_adv_values)
        mses_val_adv.append(loss_val_adv_values[-1])

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


        fig = create_example_fig(t_set["tb_image_val_indices"], zeta, dataset_val, model_d)
        writer.add_image(f"images/reconstruction", pltfig_to_tensor(fig, dpi=t_set["tb_image_dpi"]), zeta_index)
        plt.close(fig)

        writer.flush()

        zeta_index += 1

    # %%
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    ax[1 if t_set["train_make_adv"] else 0].plot(zetas, mses_train, label="mse_train")
    ax[0].plot(zetas, mses_val_std, label="mse_val_std")
    ax[1].plot(zetas, mses_val_adv, label="mse_val_adv")

    #ax.plot(zetas, predicted_mse, label="predicted")
    ax[0].set_xlabel(r"$\zeta$")
    ax[0].set_ylabel(r"$R_0$")
    ax[0].set_title(r"standard risk over $\zeta = \frac{\varepsilon^2}{\mathbb{E}[ |x|^2]}$")
    ax[1].set_xlabel(r"$\zeta$")
    ax[1].set_ylabel(r"$R_{\epsilon}$")
    ax[1].set_title(r"robust risk over $\zeta = \frac{\varepsilon^2}{\mathbb{E}[ |x|^2]}$")
    ax[0].legend()
    ax[1].legend()

    writer.add_image("images/mse_train_val", pltfig_to_tensor(fig))
    plt.close(fig)

    # %% final cleanup and remove started file
    writer.close()
    os.remove(os.path.join(log_dir, "started_run.txt"))

    return loss_train_series_per_epoch, loss_val_std_series_per_epoch, loss_val_adv_series_per_epoch

def create_example_fig(img_val_indices, zeta, dataset_val, model):
    #img_val_indices = t_set["tb_image_val_indices"]
    fig, ax = plt.subplots(nrows=len(img_val_indices), ncols=5, sharex=True, sharey=True, figsize=(4*5, 4*len(img_val_indices)))
    fig.suptitle(f"val Image examples for zeta={zeta} and indices={img_val_indices}")
    for c, img_val_index in enumerate(img_val_indices):
        X, Xorig, y = dataset_val[img_val_index]
        network_input, masked_kspace, target, _, _, _ = data_transform(X, y, return_meta_data=True) # mask, mean, std
        prediction = model(network_input.unsqueeze(0))
        meas_img = np.abs(X.squeeze().detach().cpu().numpy())
        meas_img = np.log(meas_img / np.max(meas_img) + 1e-9)
        ax[c, 0].imshow(np.linalg.norm(meas_img, axis=2), 'gray')
        meas_img = np.abs(masked_kspace.squeeze().cpu().numpy())
        meas_img = np.log(meas_img / np.max(meas_img) + 1e-9)
        ax[c, 1].imshow(np.linalg.norm(meas_img, axis=2), 'gray')
        ax[c, 2].imshow(network_input.squeeze().cpu().numpy(), 'gray')
        ax[c, 3].imshow(prediction.squeeze().detach().cpu().numpy(), 'gray')
        ax[c, 4].imshow(target.squeeze().detach().cpu().numpy(), 'gray')
    fig.tight_layout()
    return fig