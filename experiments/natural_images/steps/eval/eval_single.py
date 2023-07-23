# %%
from random import random
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
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

from generic.transforms.ApplyConvolution import ApplyConvolution, create_kernel_by_name

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

    if zeta_intersect:
        zetas = np.intersect1d(zetas, zetas_eval)
    else:
        zetas = zetas_eval

    d_set = config["Data"]
    # backwards compatability
    if test_dataset_config is None:
        print("test_dataset_config is None, try to load Data section from settings file.")
        testdir = os.path.join(d_set["path"], 'val')
        tensorlist = d_set["tensorlist"]
        if tensorlist:
            tensors = single_tensor = True
    else:
        print(f"test_dataset_config is: {test_dataset_config}")
        testdir = test_dataset_config["path_to"]
        tensors = test_dataset_config["output_format"]["tensors"]
        single_tensor = test_dataset_config["output_format"]["single_tensor"]

    if test_dataset_per_device is None:
        if tensors:
            # should work for both legacy and new method
            print(f"Using val tensorlist")
            dataset_test_base = TensorListDataset(testdir, device=device, single_tensor=single_tensor)
        else:
            # only works for new configuration method.
            from experiments.natural_images.steps.preprocessing import transform_stack_via_config
            test_dataset_config
            transform_stack = transform_stack_via_config(test_dataset_config["transform_random_crop"], test_dataset_config["transform_grayscale"], test_dataset_config["transform_normalize"])
            dataset_test_base = torchvision.datasets.ImageFolder(testdir, transform_stack)
    else:
        print(f"Test dataset was provided!")                
        dataset_test_base = test_dataset_per_device[device]

    model_version = t_set["model_version"] if "model_version" in t_set else None
    print(f"model_version is: {model_version}")
    if model_version == "V0":
        model_d = unetv0.UNet(enc_chs=(eval(t_set["unet_classes"]),) + eval(t_set["enc_chs"]), dec_chs=eval(t_set["dec_chs"]), num_class=eval(t_set["unet_classes"])).to(device)
    else:
        model_d = unetv1.UNet(enc_chs=(eval(t_set["unet_classes"]),) + eval(t_set["enc_chs"]), dec_chs=eval(t_set["dec_chs"]), num_class=eval(t_set["unet_classes"])).to(device)

    #epoch = 100
    if epoch is not None and int(t_set["epochs"]) <= epoch:
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
            model_d = model_d.to(device)
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
                'use_best': use_best
            }

            if "train_eval_conv_kernel" in t_set and t_set["train_eval_conv_kernel"] != "None":
                kernel_type = t_set["train_eval_conv_kernel"]
                kernel = create_kernel_by_name(kernel_type, channels=int(t_set["unet_classes"]))
                #print(f"Using kernel (type: {kernel_type}): {kernel} ")
                dataset_test = NoiseReconstructionDataset(dataset=dataset_test_base,
                    transform=transforms.Compose([
                        ApplyConvolution(kernel=kernel, channels=int(t_set["unet_classes"]), device=device),
                        AddGaussianNoise(mean=0, std=noise_level)]))
            else:
                dataset_test = NoiseReconstructionDataset(dataset=dataset_test_base,
                    transform=AddGaussianNoise(mean=0, std=noise_level))
            
            print(f"Noise level={noise_level}, adv_its={adv_its}, adv_random_start={adv_random_start}, adv_restarts={adv_restarts}, test_batch_size={batch_size}")
                                        
            dataset_test_loader = torch.utils.data.DataLoader(dataset_test,
                                                        batch_size=batch_size, shuffle=True,
                                                        num_workers=int(t_set['num_workers']), pin_memory=False)

            # Train network:
            if "test_loss_fn_name" in t_set:
                test_loss_fn_name = t_set["test_loss_fn_name"]
                test_loss_fn = get_loss_fn(test_loss_fn_name, eval(t_set["test_loss_fn_params"]))
            else:
                #print("No train_loss_fn_name specified. Use MSELoss.")
                test_loss_fn = get_loss_fn("mse", {})

            loss_test_epoch_t = test(dataset_test_loader, attackerModel, device, test_loss_fn, adv_test, **attack_kwargs)
            zeta_ind_true = list(zetas_eval).index(zeta)
            result_tensor[index, noise_level_ind, zeta_ind_true] = loss_test_epoch_t