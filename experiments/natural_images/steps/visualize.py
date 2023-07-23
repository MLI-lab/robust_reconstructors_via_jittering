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
import torch.nn.functional as F
from experiments.natural_images.steps.preprocessing import dataset_via_config, transform_normlizations_via_config

# %%
# takes in one configuration of a trained network and visualizes the reconstructions
def visualize_network_reconstructions(base_dir, test_dataset_config, zeta_train, zeta_test, noise_level, data_indices, output_dir, adv_its, adv_random_start = False, adv_random_restart=0, adv_random_mode="uniform_in_sphere", example_filename ="examples.png", diff_filename = "diff.png", devices = ["cpu"]):

    device = devices[0]
    dataset_test_base = dataset_via_config(test_dataset_config, device=device)

    paths = glob.glob(os.path.join(base_dir, "*", "*", "models", f"model_zeta_{zeta_train}.pt"))

    if len(paths) == 0:
        print(f"Path does not exist: {base_dir}.")

    run_dir = os.path.dirname(os.path.dirname(paths[0]))
    print(f"run dir is: {run_dir}")

    config_path = os.path.join(run_dir, "settings")
    config = configparser.ConfigParser()
    config.read(config_path)
    t_set = config["Training"]; d_set = config["Data"]

    if "train_eval_conv_kernel" in t_set and t_set["train_eval_conv_kernel"] != "None":
        kernel_type = t_set["train_eval_conv_kernel"]
        kernel = create_kernel_by_name(kernel_type, channels=int(t_set["unet_classes"]))
        dataset_test = NoiseReconstructionDataset(dataset=dataset_test_base,
            transform=transforms.Compose([
                ApplyConvolution(kernel=kernel, channels=int(t_set["unet_classes"]), device=device),
                AddGaussianNoise(mean=0, std=noise_level)]))
    else:
        dataset_test = NoiseReconstructionDataset(dataset=dataset_test_base,
            transform=AddGaussianNoise(mean=0, std=noise_level))

    _, normalization_inv = transform_normlizations_via_config(test_dataset_config["transform_normalize"])
    def tensor_to_np(t):
        return np.swapaxes(torch.squeeze(normalization_inv(t), dim=0).to("cpu").detach().numpy(), 0, 2)

    visualization_data = []
    metric_data = []
    for c, img_test_index in enumerate(data_indices):
        measurement, orig_image = dataset_test[img_test_index]
        orig_image_inp, denoisedImg, newInput, prediction, target_wc_mse, target_mse, input_mse = seek_perturbed_reconstructions(run_dir, zeta_train, zeta_test, measurement, orig_image, device, adv_its, adv_random_start, adv_random_restart, adv_random_mode)

        visualization_data.append(
            (
                tensor_to_np(measurement),
                tensor_to_np(orig_image),
                tensor_to_np(orig_image_inp),
                tensor_to_np(denoisedImg),
                tensor_to_np(newInput),
                tensor_to_np(prediction)
            )
        )
        metric_data.append((target_wc_mse, target_mse, input_mse))

    # full data visualization
    fig, ax = plt.subplots(nrows=len(data_indices), ncols=6, figsize=(6*4, 4*len(data_indices)))
    fig.suptitle(f"Examples for zeta_tr={zeta_train}, zeta_te={zeta_test} and indices={data_indices}")
    for c, img_test_index in enumerate(data_indices):
        ax_obj = ax[c] if len(data_indices) > 1 else ax
        ax_obj[0].set_title("measurement")
        ax_obj[1].set_title("orig_image")
        ax_obj[2].set_title("orig_image inp")
        ax_obj[3].set_title(f"reconstruction ({metric_data[c][1]})")
        ax_obj[4].set_title(f"perturbed input ({metric_data[c][2]})")
        ax_obj[5].set_title(f"perturbed output ({metric_data[c][0]})")

        for i in range(len(visualization_data[c])):
            ax_obj[i].imshow(visualization_data[c][i])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, example_filename))

    # diff visualization
    fig, ax = plt.subplots(nrows=len(data_indices), ncols=3, figsize=(3*4, 4*len(data_indices)))
    fig.suptitle(f"Examples for zeta_tr={zeta_train}, zeta_te={zeta_test} and indices={data_indices}")
    for c, img_test_index in enumerate(data_indices):
        ax_obj = ax[c] if len(data_indices) > 1 else ax

        ax_obj[0].set_title(f"input diff ({metric_data[c][2]})") # how does the perturbation looks like
        ax_obj[1].set_title(f"perturbed diff ({metric_data[c][0]})") # how has the perturb
        ax_obj[2].set_title(f"reconstruction diff ({metric_data[c][1]})") # how has the perturb

        ax_obj[0].imshow( np.abs(visualization_data[c][0] - visualization_data[c][4]) )
        ax_obj[1].imshow( np.abs(visualization_data[c][5] - visualization_data[c][2]) )
        ax_obj[2].imshow( np.abs(visualization_data[c][3] - visualization_data[c][2]) )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, diff_filename))

def seek_perturbed_reconstructions(base_run_dir, zeta, zeta_test, test_measurement, orig_image, device, adv_its, adv_random_start, adv_random_restart, adv_random_mode):
    config_path = os.path.join(base_run_dir, "settings")
    config = configparser.ConfigParser()
    config.read(config_path)
    d_set = config["Data"]
    t_set = config["Training"]
    print(f"Base_run_dir: {base_run_dir}")
    model_dir = os.path.join(base_run_dir, t_set["save_models_subdir"])

    model_version = t_set["model_version"] if "model_version" in t_set else None
    print(f"model_version is: {model_version}")
    if model_version == "V0":
        model_d = unetv0.UNet(enc_chs=(eval(t_set["unet_classes"]),) + eval(t_set["enc_chs"]), dec_chs=eval(t_set["dec_chs"]), num_class=eval(t_set["unet_classes"])).to(device)
    else:
        model_d = unetv1.UNet(enc_chs=(eval(t_set["unet_classes"]),) + eval(t_set["enc_chs"]), dec_chs=eval(t_set["dec_chs"]), num_class=eval(t_set["unet_classes"])).to(device)

    norm_sq = float(t_set["expected_signal_norm_sq"])
    #norm_sq = 1000000
    eps = math.sqrt(float(zeta_test) * norm_sq)

    epoch = None
    fixed_model_path = None
    filename = f"model_zeta_{zeta}.pt" if fixed_model_path is None else fixed_model_path
    if epoch is not None:
        model_path = os.path.join(model_dir, f"_{epoch}e", filename)
    else:
        model_path = os.path.join(model_dir, filename)

    model_path = os.path.join(model_dir, f"model_zeta_{zeta}.pt")
    model_dict = torch.load(model_path, map_location=device)
    model_state_dict = model_dict["model_state_dict"]
    model_d.load_state_dict(model_state_dict)
    test_unsqueezed = torch.unsqueeze(test_measurement, dim=0)
    orig_unsqueezed = torch.unsqueeze(orig_image, dim=0)
    imgDenoised = model_d(test_unsqueezed)

    attackerModel = AttackerModel(model_d)

    random_start = adv_random_start
    random_restart = adv_random_restart # is not actually used
    attack_kwargs = {
        'constraint': '2',
        'eps': eps,
        'step_size': 2.5*eps/float(adv_its), # eps / float(adv_its),
        'iterations': int(adv_its),
        'random_start': random_start,
        'random_restarts': random_restart,
        'random_mode' : adv_random_mode,
        'use_best': False
    }

    new_input, prediction = attackerModel(test_unsqueezed, target=orig_unsqueezed, make_adv=True, **attack_kwargs)

    newDistance = (new_input - test_unsqueezed).pow(2).sum().pow(0.5)
    print(f"Eps = {eps}, newDist = {newDistance}")

    if orig_image.shape != prediction.shape:
        orig_image_inp = F.interpolate(orig_unsqueezed,size=prediction.shape[-2:])
    else:
        orig_image_inp = orig_image

    target_wc_mse = torch.square(torch.squeeze(prediction) - orig_image_inp).mean().detach().cpu().item()
    target_mse = torch.square(torch.squeeze(imgDenoised) - orig_image_inp).mean().detach().cpu().item()
    input_mse = torch.square(torch.squeeze(new_input) - test_measurement).mean().detach().cpu().item()

    return orig_image_inp, imgDenoised, new_input, prediction, target_wc_mse, target_mse, input_mse

