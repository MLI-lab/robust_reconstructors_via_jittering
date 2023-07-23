import torch.multiprocessing as mp
from experiments.subspace.steps.eval.eval_single import eval_single_rundir
from generic.helper.mp_helper import distr_configs_over_gpus2
import math
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from glob import glob
import tikzplotlib as tplt

from generic.signal_models.TensorListDataset import TensorListDataset

#def average_over():
def calc_jittering(n, d, sigma_c, sigma_z, eps):
    return np.sqrt(eps**2 * sigma_z**2 * d/n + sigma_z * sigma_c * eps * math.sqrt(d/n) * np.sqrt(sigma_c**2 - eps**2 + sigma_z**2 * d/n)) / np.sqrt(d * (sigma_c**2 - eps**2))

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def run_hpsearch_eval(
    d_config_val,
    t_set_base,
    devices, method_name,
    base_path_runs,
    base_path_eval,
    rec_operator_type = "factor_zeros",
    noise_level= 0.25,
    epochs = 40,
    grid_size = 41,
    alpha_start = 0.25,
    alpha_end = 0.9,
    zetas_end = 0.3,
    zetas_steps = 11,
    smoothing_N = 7,
    reps = 4,
    adv_test = True,
    adv_its = 5,
    adv_random_start = True,
    adv_restarts = 1,
    eval_epoch_backcalc = 0
    ):

    zetas = np.linspace(0, zetas_end, zetas_steps)
    alphas = np.linspace(alpha_start, alpha_end, grid_size)

    if not os.path.exists(base_path_eval):
        os.makedirs(base_path_eval)

    #training = True
    evaluation = True
    plot = True

    batch_size = t_set_base["train_dataloader_batch_size"]

    # Evaluation
    if evaluation:
        run_dirs_list = [[] for _ in range(reps)]
        for alpha in alphas:
            path = os.path.join(base_path_runs, f"hpsearch_subspace_{method_name}_{alpha}nlt_{noise_level}nl_{epochs}e_{rec_operator_type}rot_color_adam2_{batch_size}bs")

            print(path)

            dirs = glob(os.path.join(path, "*"))
            print(dirs)
            if len(dirs) == 0:
                print(f"Path {path} led to empty dir.")
            for i in range(reps):
                run_dirs_list[i].append(dirs[i])

        print(f"run_dirs are:")
        for i in range(reps):
            print(run_dirs_list[i])

        dataset_per_device = None

        fixed_noise_levels = [noise_level]
        eval_data_full = np.zeros( (reps, eval_epoch_backcalc+1, grid_size, 1, len(zetas)))
        for i in range(reps):
            print(f"Eval for rep {i+1} of {reps}")
            run_dirs = run_dirs_list[i]

            eval_data_single_rep = np.zeros( (eval_epoch_backcalc+1, grid_size, 1, len(zetas)))

            # latest epoch
            print(f" Latest epoch models at rep{i+1}")
            tensor = torch.zeros( (len(run_dirs), len(fixed_noise_levels), len(zetas)) )
            tensor.share_memory_()
            params = []
            fixed_model_path = f"model_zeta_{0.0}.pt"
            for _, run_dir in enumerate(run_dirs):
                #for zeta in zetas:
                params.append( (run_dir, d_config_val, tensor, fixed_noise_levels, adv_test, adv_its, adv_random_start, adv_restarts, zetas, None, fixed_model_path, False, batch_size, dataset_per_device) ) # None means latest epoch here
                #break
            distr_configs_over_gpus2(devices, params, eval_single_rundir, nr_per_device=2)

            eval_data_single_rep[0, ...] = tensor.detach().cpu().numpy() # shape: (len(runs), len(noise_levels), len(zeta))
            #print(f"Got eval_data: {eval_data}")
            #return

            for j in range(eval_epoch_backcalc):
                tensor = torch.zeros( (len(run_dirs), len(fixed_noise_levels), len(zetas)) )
                tensor.share_memory_()
                params = []
                fixed_model_path = f"model_zeta_{0.0}.pt"
                epoch = epochs - j - 1
                print(f" Model at epoch {epoch} / {epochs} for  rep{i+1}")
                for _, run_dir in enumerate(run_dirs):
                    #for zeta in zetas:
                    params.append( (run_dir, tensor, fixed_noise_levels, adv_test, adv_its, adv_random_start, adv_restarts, zetas, epoch, fixed_model_path, False, batch_size, dataset_per_device) ) # None means latest epoch here
                    #break
                distr_configs_over_gpus2(devices, params, eval_single_rundir, nr_per_device=2)

                eval_data_single_rep[j+1, ...] = tensor.detach().cpu().numpy() # shape: (len(runs), len(noise_levels), len(zeta))
                
            # now we have ( len(backwards_epochs), len(run_dirs), len(fixed_noise_level), len(zetas))
            eval_data_full[i, ...] = eval_data_single_rep

        # sizes (reps,  epoch_backward_calc, runs, noise_levels, zetas), e.g. (5 x alpha_grid x 1 x zetas)
        data = np.squeeze(eval_data_full, axis=3) # noise_level dimension is 0
        print(data) # has shape (reps, backward_calc, alphas, zetas)

        # save individual repetitions
        for i in range(reps):
            df = pd.DataFrame(data[i,0,:,:])
            df.to_csv(os.path.join(base_path_eval, f"out_rep_{i}_last.csv"))
            for j in range(eval_epoch_backcalc):
                df = pd.DataFrame(data[i,j+1,:,:])
                df.to_csv(os.path.join(base_path_eval, f"out_rep_{i}_epoch_{j}.csv"))

    if plot:
        if not evaluation:
            print("load data")
            data = np.zeros( (reps, eval_epoch_backcalc+1, len(alphas), len(zetas) ))
            for i in range(reps):
                data[i,0,:,:] = pd.read_csv(os.path.join(base_path_eval, f"out_rep_{i}_last.csv")).to_numpy()[:,1:]
                for j in range(eval_epoch_backcalc):
                    data[i,j+1,:,:] = pd.read_csv(os.path.join(base_path_eval, f"out_rep_{i}_epoch_{j}.csv")).to_numpy()[:,1:]

        # save mean and std

        print(data.shape)

        data_full_last = data[:,0,:,:]
        data_full_min = np.min(data, axis=1)

        data_last_min    = np.min(data_full_last, axis=0)
        data_last_mean   = np.mean(data_full_last, axis=0)
        data_last_median = np.median(data_full_last, axis=0)
        data_last_std    = np.std(data_full_last, axis=0)

        data_min_min     = np.min(data_full_min, axis=0)
        data_min_mean    = np.mean(data_full_min, axis=0)
        data_min_median  = np.median(data_full_min, axis=0)
        data_min_std     = np.std(data_full_min, axis=0)


        if data.shape[0] > smoothing_N:
            N = smoothing_N
            filt = np.ones(N)/N
            data_min_min = np.copy(data_min_min)
            data_min_min[(N//2):-(N//2)]    = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_min_min)
            data_min_mean = np.copy(data_min_mean)
            data_min_mean[(N//2):-(N//2)]   = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_min_mean)
            data_min_median = np.copy(data_min_median)
            data_min_median[(N//2):-(N//2)] = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_min_median)
            data_min_std = np.copy(data_min_std)
            data_min_std[(N//2):-(N//2)] = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_min_std)

            N = smoothing_N
            filt = np.ones(N)/N
            data_last_min = np.copy(data_last_min)
            data_last_min[(N//2):-(N//2)]    = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_last_min)
            data_last_mean = np.copy(data_last_mean)
            data_last_mean[(N//2):-(N//2)]   = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_last_mean)
            data_last_median = np.copy(data_last_median)
            data_last_median[(N//2):-(N//2)] = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_last_median)
            data_last_std = np.copy(data_last_std)
            data_last_std[(N//2):-(N//2)] = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data_last_std)
        else:
            print("Skipping smoothing.")
        #alphas = alphas[(N//2):-(N//2)]

        # improvements
        impr_l1 = np.sum(np.abs(data_full_last - data_full_min))
        impr_l2 = np.sqrt(np.sum(np.square(data_full_last - data_full_min)))
        print(f"Abs improvements: {impr_l1}, l2 dist: {impr_l2}")

        # previous 8 objects have shape (len(alphas), len(zetas) )

        pd.DataFrame(data_last_min).to_csv(os.path.join(base_path_eval, f"min_last.csv"))
        pd.DataFrame(data_last_mean).to_csv(os.path.join(base_path_eval, f"mean_last.csv"))
        pd.DataFrame(data_last_median).to_csv(os.path.join(base_path_eval, f"median_last.csv"))
        pd.DataFrame(data_last_std).to_csv(os.path.join(base_path_eval, f"std_last.csv"))

        pd.DataFrame(data_min_min).to_csv(os.path.join(base_path_eval, f"min_min.csv"))
        pd.DataFrame(data_min_mean).to_csv(os.path.join(base_path_eval, f"mean_min.csv"))
        pd.DataFrame(data_min_median).to_csv(os.path.join(base_path_eval, f"median_min.csv"))
        pd.DataFrame(data_min_std).to_csv(os.path.join(base_path_eval, f"std_min.csv"))

        d = d_config_val["d"]
        n = d_config_val["n"]
        plot_step(base_path_eval, zetas, alphas, d,n, noise_level, method_name,
            data_last_min, data_last_mean, data_last_median, data_last_std, "last")
        plot_step(base_path_eval, zetas, alphas, d,n, noise_level, method_name,
            data_min_min, data_min_mean, data_min_median, data_min_std, "min")

def plot_step(base_path_jit_unet_eval, zetas, alphas, n, d, noise_level, method_name, data_min, data_mean, data_median, data_std, label):
    # full plot
    fig, ax = plt.subplots(ncols=1, nrows=1)
    mean_minimizers = np.zeros(len(zetas))
    min_minimizers = np.zeros(len(zetas))
    for zeta_ind, zeta in enumerate(zetas):
        data_mean_loc = data_mean[:,zeta_ind]
        data_min_loc = data_min[:,zeta_ind]
        data_std_loc = data_std[:,zeta_ind]
        ax.plot(alphas, data_mean_loc, label=r"$\zeta$=" + f"{zeta:.2f}")
        data_lb = data_mean_loc - data_std_loc
        data_ub = data_mean_loc + data_std_loc
        ax.fill_between(alphas, data_lb, data_ub, alpha=0.2)#, label=r"$\zeta$=" + f"{zeta:.2f}")
        mean_minimizers[zeta_ind] = alphas[np.argmin(data_mean_loc)]
        min_minimizers[zeta_ind] = alphas[np.argmin(data_min_loc)]
    ax.set_xlabel("reg param")
    ax.set_ylabel("robust risk")
    fig.legend()
    fig.tight_layout()
    tikzplotlib_fix_ncols(fig)
    tplt.save(os.path.join(base_path_jit_unet_eval, f"plot_{label}.tikz"))
    fig.savefig(os.path.join(base_path_jit_unet_eval, f"plot_{label}.png"))

    fig, ax = plt.subplots(ncols=1, nrows=1)
    for zeta_ind, zeta in enumerate(zetas):
        data_median_loc = data_median[:,zeta_ind]
        ax.plot(alphas, data_median_loc, label=r"$\zeta$=" + f"{zeta:.2f}")
    ax.set_xlabel("jittering noise")
    ax.set_ylabel("robust risk")
    fig.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_path_jit_unet_eval, f"median_plot_{label}.png"))
    tikzplotlib_fix_ncols(fig)
    tplt.save(os.path.join(base_path_jit_unet_eval, f"median_plot_{label}.tikz"))

    fig, ax = plt.subplots(ncols=1, nrows=1)
    for zeta_ind, zeta in enumerate(zetas):
        data_min_loc = data_min[:,zeta_ind]
        ax.plot(alphas, data_min_loc, label=r"$\zeta$=" + f"{zeta:.2f}")
        #ax.scatter(alphas, data_min_loc, label=r"$\zeta$=" + f"{zeta:.2f}")
        ind = np.argmin(data_min_loc)
        ax.plot(alphas[ind], data_min_loc[ind], "*", label=r"$\zeta$=" + f"{zeta:.2f}")
    ax.set_xlabel("jittering noise")
    ax.set_ylabel("robust risk")
    fig.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_path_jit_unet_eval, f"min_{label}.png"))
    tikzplotlib_fix_ncols(fig)
    tplt.save(os.path.join(base_path_jit_unet_eval, f"min_{label}.tikz"))

    pd.DataFrame(mean_minimizers).to_csv(os.path.join(base_path_jit_unet_eval, f"mean_minimizers_{label}.csv"))
    pd.DataFrame(min_minimizers).to_csv(os.path.join(base_path_jit_unet_eval, f"min_minimizers_{label}.csv"))

    if method_name == "jittering":
        sigma_c = math.sqrt(d)
        zetas_full = np.linspace(np.min(zetas), np.max(zetas), 100)
        ones_np = np.ones_like(zetas_full)
        xi_test = noise_level;
        sigma_z = xi_test * math.sqrt(n); eps = np.sqrt(zetas_full) * sigma_c
        alpha_pred = calc_jittering(n, d, ones_np*sigma_c, ones_np*sigma_z, eps)
        alpha_pred = np.sqrt(alpha_pred**2 + noise_level**2)

    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(zetas, mean_minimizers, label=f"mean")
    ax.plot(zetas, min_minimizers, label=f"min")
    if method_name == "jittering":
        ax.plot(zetas_full, alpha_pred, label=f"Prediction")
    ax.set_xlabel("zeta")
    ax.set_ylabel("alpha min")
    fig.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_path_jit_unet_eval,f"mean_min_{label}.png"))
    tikzplotlib_fix_ncols(fig)
    tplt.save(os.path.join(base_path_jit_unet_eval, f"mean_min_{label}.tikz"))

    fig, ax = plt.subplots(ncols=1, nrows=1)
    if method_name == "jittering":
        ax.plot(zetas_full, alpha_pred, label=f"Prediction")
    for min_ind, min in enumerate(min_minimizers):
        ax.plot(zetas[min_ind], min,".", label=r"$\zeta$=" + f"{zetas[min_ind]:.2f}")
    ax.set_xlabel("zeta")
    ax.set_ylabel("alpha min")
    fig.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_path_jit_unet_eval, f"mean_min_scatter_{label}.png"))
    tikzplotlib_fix_ncols(fig)
    tplt.save(os.path.join(base_path_jit_unet_eval, f"mean_min_scatter_{label}.tikz"))
