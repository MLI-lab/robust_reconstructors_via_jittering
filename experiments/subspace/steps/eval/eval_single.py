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

from generic.reconstruction_models.SubspaceProjectionNetwork import NeuralNetwork

from experiments.subspace.steps.preprocessing import (
    dataset_via_config
)

from generic.transforms.LinearTransformation import LinearTransformation, create_diagonal_by_name

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

    #epoch = 100
    if epoch is not None and int(t_set["epochs"]) <= epoch:
        epoch = None

    n = test_dataset_config["n"]
    d = test_dataset_config["d"]

    not_skip_linear_diagonal = eval(t_set["not_skip_linear_diagonal"]) if "not_skip_linear_diagonal" in t_set else False
    full_linear = eval(t_set["full_linear"]) if "full_linear" in t_set else False
    model_d = NeuralNetwork(n, d, device, eval(t_set["skip_linear"]), not_skip_linear_diagonal, full_linear).to(device)

    for noise_level_ind, noise_level in enumerate(fixed_noise_levels):
        for zeta_ind, zeta in enumerate(zetas):

            norm_sq = float(d)
            eps = math.sqrt(float(zeta) * norm_sq)

            print(f"Evalute at zeta={zeta} with eps={eps}, n={n} and d={d}")
            print(f"Fixed model path is: {fixed_model_path}")
            filename = f"model_zeta_{zeta}.pt" if fixed_model_path is None else fixed_model_path
            if epoch is not None:
                model_path = os.path.join(model_dir, f"_{epoch}e", filename)
            else:
                model_path = os.path.join(model_dir, filename)


            model_dict = torch.load(model_path, map_location=device)
            model_state_dict = model_dict["model_state_dict"]
            model_d.load_state_dict(model_state_dict)

            attackerModel = AttackerModel(model_d)
            adv_its = float(adv_its)
            attack_kwargs = {
                'constraint': '2',
                'eps': eps,
                'step_size': 2.5*eps / float(adv_its),
                'iterations': int(adv_its),
                'random_start': adv_random_start,
                'random_restarts': adv_restarts,
                'use_best': False
            }

            print(f"Eval at test, zeta={zeta} with noise_level={noise_level} make_adv={adv_test} and adv_its={adv_its}")

            dataset_test = dataset_via_config(test_dataset_config, device=device, noise_level=noise_level)

            dataset_test_loader = torch.utils.data.DataLoader(dataset_test,
                                                        batch_size=batch_size, shuffle=False,
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