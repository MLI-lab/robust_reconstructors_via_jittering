# %%
import configparser
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import generic.reconstruction_models.UNetReconstruction as unetv1
import generic.reconstruction_models.UNetReconstructionV0 as unetv0
from generic.signal_models.NoiseReconstructionDataset import NoiseReconstructionDataset
from generic.robust_training.AttackerModel import AttackerModel
from generic.transforms.AddGaussianNoise import AddGaussianNoise
from generic.signal_models.TensorListDataset import TensorListDataset
from generic.transforms.AddGaussianNoise import AddGaussianNoise
from generic.transforms.ApplyConvolution import ApplyConvolution, create_kernel_by_name
from torchvision.transforms.functional import rotate
from generic.signal_models.MRITransformedDataset import MRITransformedDataset
import torch.nn.functional as F
from experiments.mri.steps.preprocessing import dataset_via_config
from generic.transforms.mri_transform import data_transform

import fastmri
import fastmri.models

def visualize_network_reconstructions(base_dir, test_dataset_config, zeta_train, zeta_test, noise_level, data_indices, output_dir, adv_its, adv_random_start = False, adv_random_restart=0, adv_random_mode="uniform_in_sphere", example_filename ="examples.png", diff_filename = "diff.png", devices = ["cpu"]):

    device = devices[0]
    dataset_test_base = dataset_via_config(test_dataset_config, device=device)

    # search the model which has been trained on zeta_train perturbation level
    paths = glob.glob(os.path.join(base_dir, "*", "*", "models", f"model_zeta_{zeta_train}.pt"))

    run_dir = os.path.dirname(os.path.dirname(paths[0]))

    transform = AddGaussianNoise(mean=0, std=noise_level) if noise_level != 0.0 else None
    dataset_test = MRITransformedDataset(dataset=dataset_test_base, transform=transform, device=device)

    def img_tensor_to_np(t):
        return torch.squeeze(torch.squeeze(t, dim=0)).to("cpu").detach().numpy()

    def kspace_tensor_to_np(t):
        meas_img = np.abs(t.squeeze().detach().cpu().numpy())
        meas_img = np.log(meas_img / np.max(meas_img) + 1e-9)
        return np.linalg.norm(meas_img, axis=2)

    visualization_data = []
    metric_data = []
    for c, img_test_index in enumerate(data_indices):

        measurement,_, target = dataset_test[img_test_index]
        network_input, masked_kspace, target_torch, mask, mean, std = data_transform(measurement, target, return_meta_data=True)

        #measurement, orig_image = dataset_test[img_test_index]
        orig_input, unperturbed_output, perturbed_input, perturbed_output, target_wc_mse, target_mse, input_mse = seek_perturbed_reconstructions(run_dir, zeta_train, zeta_test, measurement, mask, mean, std, target, device, adv_its, adv_random_start, adv_random_restart, adv_random_mode)

        network_input_perturbed, _ = data_transform(perturbed_input, target, fixed_mask=mask, fixed_mean=mean, fixed_std=std, return_meta_data=False)

        visualization_data.append(
            (
                kspace_tensor_to_np(measurement), #0
                img_tensor_to_np(target_torch), #1
                kspace_tensor_to_np(masked_kspace), #2
                img_tensor_to_np(network_input), #3
                img_tensor_to_np(unperturbed_output), #4
                kspace_tensor_to_np(perturbed_input), #5
                img_tensor_to_np(network_input_perturbed), #6
                img_tensor_to_np(perturbed_output), #7
                kspace_tensor_to_np(torch.abs(measurement-perturbed_input)) #8
            )
        )
        metric_data.append((target_wc_mse, target_mse, input_mse))

    # full data visualization
    fig, ax = plt.subplots(nrows=len(data_indices), ncols=8, figsize=(8*4, 4*len(data_indices)))
    fig.suptitle(f"Examples for zeta_tr={zeta_train}, zeta_te={zeta_test} and indices={data_indices}")
    for c, img_test_index in enumerate(data_indices):
        ax_obj = ax[c] if len(data_indices) > 1 else ax
        ax_obj[0].set_title("measurement")
        ax_obj[1].set_title(f"target")
        ax_obj[2].set_title("masked measurement")
        ax_obj[3].set_title(f"network input")
        ax_obj[4].set_title("unperturbed output")
        ax_obj[5].set_title("perturbed measurement")
        ax_obj[6].set_title(f"perturbed network input")
        ax_obj[7].set_title("perturbed output")

        for i in range(len(visualization_data[c])-1):
            ax_obj[i].imshow(visualization_data[c][i], cmap='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, example_filename))

    # diff visualization
    fig, ax = plt.subplots(nrows=len(data_indices), ncols=5, figsize=(5*4, 4*len(data_indices)))
    fig.suptitle(f"Examples for zeta_tr={zeta_train}, zeta_te={zeta_test} and indices={data_indices}")
    for c, img_test_index in enumerate(data_indices):
        ax_obj = ax[c] if len(data_indices) > 1 else ax
        ax_obj[0].set_title(f"measurement pert. diff. ({metric_data[c][2]:.4f})")
        ax_obj[1].set_title(f"network input diff.")
        ax_obj[2].set_title(f"unperturbed output vs target ({metric_data[c][1]:.4f})")
        ax_obj[3].set_title(f"perturbed output vs target ({metric_data[c][0]:.4f})")
        ax_obj[4].set_title("unperturbed vs perturbed output")

        ax_obj[0].imshow(visualization_data[c][8], cmap='gray')
        ax_obj[1].imshow(np.abs(visualization_data[c][3]-visualization_data[c][6]), cmap='gray')
        ax_obj[2].imshow(np.abs(visualization_data[c][4]-visualization_data[c][1]), cmap='gray')
        ax_obj[3].imshow(np.abs(visualization_data[c][7]-visualization_data[c][1]), cmap='gray')
        ax_obj[4].imshow(np.abs(visualization_data[c][7]-visualization_data[c][4]), cmap='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, diff_filename))

def seek_perturbed_reconstructions(base_run_dir, zeta, zeta_test, test_measurement, mask, mean, std, target, device, adv_its, adv_random_start, adv_random_restart, adv_random_mode):
    config_path = os.path.join(base_run_dir, "settings")
    config = configparser.ConfigParser()
    config.read(config_path)
    #d_set = config["Data"]
    t_set = config["Training"]
    print(f"Base_run_dir: {base_run_dir}")
    model_dir = os.path.join(base_run_dir, t_set["save_models_subdir"])

    norm_sq = float(t_set["expected_signal_norm_sq"])
    eps = math.sqrt(float(zeta_test) * norm_sq)

    filename = f"model_zeta_{zeta}.pt" #if fixed_model_path is None else fixed_model_path
    model_path = os.path.join(model_dir, filename)

    model_d = fastmri.models.Unet(in_chans=1, out_chans=1, chans=16, num_pool_layers=4, drop_prob=0.0).to(device)

    model_dict = torch.load(model_path, map_location=device)
    model_state_dict = model_dict["model_state_dict"]
    model_d.load_state_dict(model_state_dict)
    #model_d = model_d.to(device)
    attackerModel = AttackerModel(model_d)
    factor = 2.5

    test_measurement = test_measurement.unsqueeze(0)
    target = target.unsqueeze(0)

    network_input, _ = data_transform(test_measurement, None, return_meta_data=False, fixed_mask=mask, fixed_mean=mean, fixed_std=std)
    prediction_orig = model_d(network_input)

    attackerModel = AttackerModel(model_d)

    def data_transform_fixed_mask(X, y):
        return data_transform(X, y, return_meta_data=False, fixed_mask=mask, fixed_mean=mean, fixed_std=std)

    print(f"Eps: {eps}")
    attack_kwargs = {
        'constraint': '2',
        'eps': eps,
        'step_size': factor*eps/float(adv_its), # eps / float(adv_its),
        'iterations': int(adv_its),
        'random_start': adv_random_start,
        'random_restarts': adv_random_restart,
        'random_mode' : adv_random_mode,
        'use_best': False,
        'data_transform' : data_transform_fixed_mask
    }

    new_input, prediction = attackerModel(test_measurement, target, make_adv=True, **attack_kwargs)

    newDistance = (new_input - test_measurement).pow(2).sum().pow(0.5)
    print(f"Eps = {eps}, newDist = {newDistance}")

    target_wc_mse = torch.square(torch.squeeze(prediction) - target).mean().detach().cpu().item()
    target_mse = torch.square(torch.squeeze(prediction_orig) - target).mean().detach().cpu().item()
    input_mse = torch.square(torch.squeeze(new_input) - test_measurement).sum().detach().cpu().item()

    return test_measurement, prediction_orig, new_input, prediction, target_wc_mse, target_mse, input_mse /  norm_sq

