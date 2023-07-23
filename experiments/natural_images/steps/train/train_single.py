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
from experiments.natural_images.steps.preprocessing import transform_stack_via_config

from experiments.natural_images.steps.train_and_test import test, train
from generic.transforms.AddGaussianNoise import AddGaussianNoise
from generic.transforms.ApplyConvolution import ApplyConvolution, create_kernel_by_name

from experiments.natural_images.steps.preprocessing import (
    dataset_via_config,
    transform_normlizations_via_config
)

import configparser
import os
from datetime import datetime

#def add_aux_entries_to_dicts(d_set, t_set):
    ## add auxilary entries
    #d_set["c"] = 1 if d_set["grayscale"] else 3
    #d_set["n"] = d_set["w"] * d_set["h"] * d_set["c"]
    #d_set["expected_signal_norm_sq"] = d_set["expected_signal_norm_sq_gray"] if d_set["grayscale"] else d_set["expected_signal_norm_sq_full"]
    #t_set["unet_classes"] = d_set["c"]
    #t_set["zetas"] = np.linspace(t_set["zetas_start"], t_set["zetas_end"], t_set["zetas_steps"])

def setup_logdir(t_set, base_artifact_path):

    trainInfo = f"train{'Adv' if t_set['train_make_adv'] else 'Std'}{t_set['train_noise_level']}nl_{t_set['train_dataloader_batch_size']}bs"
    testInfo = f"test{'Adv' if t_set['test_make_adv'] else 'Std'}{t_set['val_noise_level']}nl_{t_set['test_dataloader_batch_size']}bs"

    #n = d_set["n"]
    expected_signal_norm_sq = t_set["expected_signal_norm_sq"]
    now = datetime.now()
    timestamp_now = f"{now.year}-{now.month}-{now.day}__{now.hour}-{now.minute}-{now.second}"
    #base_artifact_path = t_set["base_artifact_path"] if "base_artifact_path" in t_set else "runs"
    #log_dir = f"runs/{t_set['tb_subfolder']}/{os.path.basename(__file__)}_{d_set['info']}_{trainInfo}_{testInfo}_{t_set['unet_classes']}c_{t_set['dec_chs'][0]}f_{n}n_{expected_signal_norm_sq}esns_{'full' if d_set['imagenet_full'] else 'red'}_{t_set['zetas_start']:.5f}-{t_set['zetas_end']:.5f}-{t_set['zetas_steps']}z_{t_set['epochs']}e_{timestamp_now}"
    log_dir = f"{base_artifact_path}/{t_set['tb_subfolder']}/{os.path.basename(__file__)}_{t_set['fixed_gpu_device']}_{trainInfo}_{testInfo}_{t_set['unet_classes']}c_{t_set['dec_chs'][0]}f_{expected_signal_norm_sq}esns_{t_set['zetas_start']:.5f}-{t_set['zetas_end']:.5f}-{t_set['zetas_steps']}z_{t_set['epochs']}e_{timestamp_now}"
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

def run_imagenet_unet_robust_reconstruction(d_config_train, d_config_val, t_set, base_artifact_path):

    #add_aux_entries_to_dicts(d_set, t_set)
    log_dir = setup_logdir(t_set, base_artifact_path)

    print(f"Setup log_dir: {log_dir}")
    save_config(log_dir, d_config_train, d_config_val, t_set) 
    create_temp_file(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # %% load device
    device = get_device(t_set)

    loss_train_series_per_epoch = []
    loss_test_series_per_epoch = []
    mses_test = []
    mses_train = []

    t_set["zetas"] = np.linspace(t_set["zetas_start"], t_set["zetas_end"], t_set["zetas_steps"])
    zetas = t_set["zetas"]

    dataset_train_base = dataset_via_config(d_config_train, device=device)
    dataset_val_base = dataset_via_config(d_config_val, device=device)

    zeta_index = 0
    for zeta in tqdm(zetas, position=0):
        sig_energy = float(t_set["expected_signal_norm_sq"])
        eps = math.sqrt(zeta * sig_energy)

        dataloader_transf = []
        if "train_eval_conv_kernel" in t_set and t_set["train_eval_conv_kernel"] != "None":
            kernel_type = t_set["train_eval_conv_kernel"]
            kernel = create_kernel_by_name(kernel_type, channels=t_set["unet_classes"])
            print(f"Using kernel (type: {kernel_type}): {kernel} ")
            dataloader_transf = [ApplyConvolution(kernel=kernel, channels=t_set["unet_classes"], device=device)]

        dataset_train = NoiseReconstructionDataset(dataset=dataset_train_base,
            transform=transforms.Compose(dataloader_transf + [AddGaussianNoise(mean=0, std=t_set["train_noise_level"])]))
        dataset_test = NoiseReconstructionDataset(dataset=dataset_val_base,
            transform=transforms.Compose(dataloader_transf + [AddGaussianNoise(mean=0, std=t_set["val_noise_level"])]))

        dataset_train_loader = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=t_set['train_dataloader_batch_size'], shuffle=True,
                                                    num_workers=t_set['num_workers'], pin_memory=t_set['train_pin_memory'])
                                        
        dataset_test_loader = torch.utils.data.DataLoader(dataset_test,
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

        model_d = UNet(enc_chs=(t_set["unet_classes"],) + t_set["enc_chs"] , dec_chs=t_set["dec_chs"], num_class=t_set["unet_classes"]).to(device)
    
        if t_set["data_parallel"]:
            model_d = torch.nn.DataParallel(model_d, device_ids=t_set["data_parallel_list"])

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
            'random_mode' : t_set["adv_random_mode"]
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
            elif scheduler_key == "None":
                pass
            else:
                print(f"Unsupported scheduler: {scheduler_key}")

        loss_train_values = []
        loss_test_values = []
    
        epochs = t_set["epochs"]

        for t in trange(epochs, desc=f"Training epochs for zeta={zeta}", unit="epoch", position=1):

            loss_train_epoch_t = train(dataset_train_loader, attackerModel, device, train_loss_fn, optimizer, t_set["train_make_adv"], **attack_kwargs)
            if scheduler != None:
                scheduler.step()

            writer.add_scalars(f"Loss/train",  {
                f"zeta={zeta}" : loss_train_epoch_t}, t)
            loss_train_values.append(loss_train_epoch_t)
            loss_test_epoch_t = test(dataset_test_loader, attackerModel, device, test_loss_fn, t_set["test_make_adv"], **attack_kwargs)
            writer.add_scalars(f"Loss/test",  {
                f"zeta={zeta}" : loss_test_epoch_t}, t)
            loss_test_values.append(loss_test_epoch_t)

            if "save_models_epochs" in t_set and t_set["save_models_epochs"]:
                if t % int(t_set["save_models_epochs_mod"]) == 0:
                    model_dir = os.path.join(log_dir, t_set['save_models_subdir'], f"_{t}e")
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    path = os.path.join(model_dir, f"model_zeta_{zeta}.pt")
                    torch.save({
                        'model_state_dict': model_d.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)
                
        # Save values:
        loss_train_series_per_epoch.append(loss_train_values)
        mses_train.append(loss_train_values[-1])
        loss_test_series_per_epoch.append(loss_test_values)
        mses_test.append(loss_test_values[-1])

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

        # Plot examplary test images
        img_test_indices = t_set["tb_image_test_indices"]

        # load inverse of normalization in dataset to show images
        _, normalization_inv = transform_normlizations_via_config(d_config_val["transform_normalize"])

        fig, ax = plt.subplots(nrows=len(img_test_indices), ncols=3, sharex=True, sharey=True, figsize=(12, 4*len(img_test_indices)))

        fig.suptitle(f"Test Image examples for zeta={zeta} and indices={img_test_indices}")
        for c, img_test_index in enumerate(img_test_indices):
            imgNoise, imgOrig = dataset_test[img_test_index]
            imgDenoised = model_d(torch.unsqueeze(imgNoise, dim=0))

            imgNoise =    normalization_inv(imgNoise)
            imgOrig =     normalization_inv(imgOrig)
            imgDenoised = normalization_inv(imgDenoised)

            imgNoise = transforms.RandomInvert(p=1)(imgNoise)
            imgOrig = transforms.RandomInvert(p=1)(imgOrig)
            imgDenoised = transforms.RandomInvert(p=1)(imgDenoised)

            ax[c,0].imshow(np.swapaxes(imgOrig.to("cpu").numpy(), 0, 2), cmap="gray", vmin=0, vmax=1)
            ax[c,1].imshow(np.swapaxes(imgNoise.to("cpu").numpy(), 0, 2), cmap="gray", vmin=0, vmax=1)
            ax[c,2].imshow(np.swapaxes(torch.squeeze(imgDenoised, dim=0).to("cpu").detach().numpy(), 0, 2), cmap="gray", vmin=0, vmax=1)

        fig.tight_layout()

        writer.add_image("images/denoising", pltfig_to_tensor(fig, dpi=t_set["tb_image_dpi"]), zeta_index)
        plt.close(fig)
        writer.flush()

        zeta_index += 1

    # %%
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))

    ax.plot(zetas, mses_train, label="exp_train")
    ax.plot(zetas, mses_test, label="exp_test")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel("mse")
    ax.set_title(r"MSE over $\zeta = \frac{\varepsilon^2}{\mathbb{E}[ |x|^2]}$")
    ax.legend()

    writer.add_image("images/mse_train_test", pltfig_to_tensor(fig))
    plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    epochs_indices = np.arange(epochs)
    for index, series_in_epoch in enumerate(loss_train_series_per_epoch):
        ax[0].plot(epochs_indices, series_in_epoch, label=r"$\zeta$=" + f"{zetas[index]}")

    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("mse")
    ax[0].set_title("Training")
    ax[0].grid()
    ax[0].legend()

    for index, series_in_epoch in enumerate(loss_test_series_per_epoch):
        ax[1].plot(epochs_indices, series_in_epoch, label=r"$\zeta$=" + f"{zetas[index]}")

    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("mse")
    ax[1].set_title("Test")
    ax[1].grid()
    ax[1].legend()

    writer.add_image("images/training_metrics", pltfig_to_tensor(fig))
    plt.close(fig)

    # %% final cleanup and remove started file
    writer.close()
    os.remove(os.path.join(log_dir, "started_run.txt"))

    return loss_train_series_per_epoch, loss_test_series_per_epoch