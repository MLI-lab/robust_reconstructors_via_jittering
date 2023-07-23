# %%
import configparser
import os
from dataclasses import dataclass
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
import pandas as pd
from generic.helper.pandas_helper import add_settings_info_to_df
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tikzplotlib as tplt

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

def _create_dirs(filename):
    if not os.path.isdir(filename):
        dir = os.path.dirname(filename)
        if not os.path.isdir(dir):
            os.makedirs(dir)
    return filename

def _save_create_tikz_fig(fig, save, create_tikz, name, dpi):
    if (save):
        if (create_tikz):
            tplt.save(f"{name}.tikz")
            print(f"\tTikzFigure saved at: {name}.tikz")
        else:
            fig.savefig(_create_dirs(f"{name}.png"), dpi=(dpi))
            print(f"\tFigure saved at: {name}.png")

def gen_jittering_std_adv_comparision_aggregatedl2errReg(base_dir_reference, base_dir_std, base_dir_stdl2, base_dir_std_err_reg, base_dir_adv,
    eval_df_save_dir_reference, eval_df_save_dir_std, eval_df_save_dir_stdl2, eval_df_save_dir_stdErrorReg, eval_df_save_dir_adv,
    noise_levels_reference, noise_levels_std, noise_levels_stdl2, noise_levels_std_err_reg, noise_levels_adv,
    df_reference_filename, df_std_filename, df_std_l2_filename, df_std_err_reg_filename, df_adv_filename,
    global_figsize=(8,6), global_dpi=100, global_create_tikz = False, global_plots_show=False, overwrite_global = False,
    plot_figsize = (8,6), plot_show = False, plot_save = False, plot_filename = "plot",
    plot_custom_colors = True,
    plot_selected_colors = ['black', 'purple', 'orange', 'red', 'green'],
    plot_linestyles = ['-', '--'], plot_ylabel="label",
    plot_create_tikz = False,
    base_output_dir = "pub",

    reference_show = True, reference_label = "reference. Tr.",
    reference_param2_restricted = True, reference_param2_selection = [0.5, 1.0],
    reference_param1_restricted = True, reference_param1_selection = [0.5, 1.0],
    reference_param1_colors = True, reference_param1_key = "sigma_tr", reference_param2_key = "sigma_te",
    reference_param1_label="sigma_tr", reference_param2_label="sigma_te",
    reference_param1_fixed_color=True, reference_param1_fixed_color_selection="red",
    reference_param1_legend_aggregated = True, reference_param1_legend_aggregated_select = True,
    reference_gather_zetas = True,

    std_show = True, std_label = "Std. Tr.",
    std_param2_restricted = True, std_param2_selection = [0.5, 1.0],
    std_param1_restricted = True, std_param1_selection = [0.5, 1.0],
    std_param1_colors = True, std_param1_key = "sigma_tr", std_param2_key = "sigma_te",
    std_param1_label="sigma_tr", std_param2_label="sigma_te",
    std_param1_fixed_color=True, std_param1_fixed_color_selection="red",
    std_param1_legend_aggregated = True, std_param1_legend_aggregated_select = True,
    std_gather_zetas = True,

    std_l2_show = True, std_l2_label = "Std. Tr. L2",
    std_param2l2_restricted = True, std_param2l2_selection = [0.5, 1.0],
    std_param1l2_restricted = True, std_param1l2_selection = [0.5, 1.0],
    std_param1l2_colors = True, std_param1l2_key = "sigma_tr", std_param2l2_key = "sigma_te",
    std_param1l2_label="sigma_tr", std_param2l2_label="sigma_te",
    std_param1l2_fixed_color=True, std_param1l2_fixed_color_selection="red",
    std_param1l2_legend_aggregated = True,std_param1l2_legend_aggregated_select = True,
    std_l2_gather_zetas = True,

    std_err_reg_show = True, std_err_reg_label = "Std. Tr. ErrReg",
    std_param2_err_reg_restricted = True, std_param2_err_reg_selection = [0.5, 1.0],
    std_param1_err_reg_restricted = True, std_param1_err_reg_selection = [0.5, 1.0],
    std_param1_err_reg_colors = True, std_param1_err_reg_key = "sigma_tr", std_param2_err_reg_key = "sigma_te",
    std_param1_err_reg_label="sigma_tr", std_param2_err_reg_label="sigma_te",
    std_param1_err_reg_fixed_color=True, std_param1_err_reg_fixed_color_selection="red",
    std_param1_err_reg_legend_aggregated = True,std_param1_err_reg_legend_aggregated_select = True,
    std_err_reg_gather_zetas = True,

    adv_show = True, adv_aggregate = False, adv_aggregate_select = False,
    adv_param2_restricted = True, adv_param2_selection = [0.5, 1.0],
    adv_param1_restricted = True, adv_param1_selection = [0.5, 1.0],
    adv_param1_colors = True, adv_param1_key = "sigma_tr", adv_param2_key = "sigma_te",
    adv_param1_label="sigma_tr", adv_param2_label="sigma_te", adv_param2_legend_hide = True
    ):


    if(len(glob(base_dir_adv)) == 0):
        print(f"No directories found at base dir: {base_dir_adv}")
        return
    zetas_adv = _read_zetas_from_first_rundir(glob(base_dir_adv)[0], True)
    print(f"zets_adv: {zetas_adv}")
    zetas_adv_columns = [f"zeta={zeta:.2}" for zeta in zetas_adv]

    if reference_show:
        if(len(glob(base_dir_reference)) == 0):
            print(f"No directories found at base dir: {base_dir_reference}")
            return

        print("Base dir reference: {glob(base_dir_reference)[0]}")
        zetas_reference = _read_zetas_from_first_rundir(glob(base_dir_reference)[0], True)
        print(f"zetas_reference: {zetas_reference}")
        zetas_reference_columns = [f"zeta={zeta:.2}" for zeta in zetas_reference]

    if(std_show):
        if(len(glob(base_dir_std)) == 0):
            print(f"No directories found at base dir: {base_dir_std}")
            return
        zetas_std = _read_zetas_from_first_rundir(glob(base_dir_std)[0], True)
        print(f"zetas_std: {zetas_std}")
        zetas_std_columns = [f"zeta={zeta:.2}" for zeta in zetas_std]

    if(std_l2_show):
        if(len(glob(base_dir_stdl2)) == 0):
            print(f"No directories found at base dir: {base_dir_stdl2}")
            return
        zetas_stdl2 = _read_zetas_from_first_rundir(glob(base_dir_stdl2)[0],True)
        print(f"zetas_stdl2: {zetas_stdl2}")
        zetas_stdl2_columns = [f"zeta={zeta:.2}" for zeta in zetas_stdl2]

    if(std_err_reg_show):
        if(len(glob(base_dir_std_err_reg)) == 0):
            print(f"No directories found at base dir: {base_dir_std_err_reg}")
            return
        zetas_stdErrReg = _read_zetas_from_first_rundir(glob(base_dir_std_err_reg)[0],True)
        print(f"zetas_stdErrReg: {zetas_stdErrReg}")
        zetas_std_err_reg_columns = [f"zeta={zeta:.2}" for zeta in zetas_stdErrReg]

    perc = 66
    zetas = zetas_adv
    print(f"zetas-adv: {zetas}")

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(plot_figsize))
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(plot_ylabel)

    colors = (plot_selected_colors)
    plot_linestyles = (plot_linestyles)


    rep_count_l2 = rep_count_errReg = rep_count_jittering = rep_count_ref = 0

    if not (overwrite_global):
        plot_figsize = global_figsize
        plot_show = global_plots_show
        plot_create_tikz = global_create_tikz

    # 0) load reference
    if (reference_show):
        print(eval_df_save_dir_reference)
        df_full_path = os.path.join(eval_df_save_dir_reference, df_reference_filename)
        if not os.path.exists(df_full_path):
            print(f"The file {df_full_path} does not exist.")
            return
        df = pd.read_csv(os.path.join(eval_df_save_dir_reference, df_reference_filename))

        dsm = {"n" : "n"}
        tsm = {"train_noise_level" : "sigma_tr", "SGD_weight_decay" : "SGD_weight_decay",
             "zetas_start" : "zetas_start"}

        add_settings_info_to_df(df, t_set_mapping=tsm)
        #df["sigma_tr"] = df["train_eps_constant"].astype(np.float32) / df["n"].astype(np.float32).pow(0.5)
        df_red = df
        #df_red = df.drop(["train_eps_constant"], axis=1)
        reference_param1_key = "zetas_start"
        df_red.sort_values(by=reference_param1_key, inplace=True)

        #print(df_red)
    
        if (reference_param1_restricted):
            param1_levels = (reference_param1_selection)
        else:
            param1_levels = np.unique(df_red[reference_param1_key])#.astype(np.float32).to_numpy()

        print(f"param1_levels: {param1_levels}")

        df_rows_for_param1_level = []
        for level in param1_levels:
            #print(f"level={level}")
            df_rows_for_param1_level.append(df_red[df_red[reference_param1_key] == level])

        values = np.zeros_like(zetas_reference)
        values_up = np.zeros_like(zetas_reference)
        values_lp = np.zeros_like(zetas_reference)
        for level_index, level in enumerate(param1_levels):
            level = float(level)
            values_per_level_index = []
            for index, row in df_rows_for_param1_level[level_index].iterrows():
                param2 = row[reference_param2_key]
                if (reference_param2_restricted):
                    if param2 not in (reference_param2_selection):
                        continue
                mse_over_zeta = row[zetas_reference_columns].to_numpy()

                mse_over_zeta = mse_over_zeta.astype(np.float32)

                if((reference_param1_legend_aggregated)):
                    if (reference_param1_legend_aggregated_select):
                        values_per_level_index.append(mse_over_zeta[level_index])
                    else:
                        values[level_index] = np.mean(mse_over_zeta)
                        values_lp[level_index] = values[level_index] - np.std(mse_over_zeta)
                        values_up[level_index] = values[level_index] + np.std(mse_over_zeta)
                else:
                    label = f"{reference_param1_label}" + f"={level:.2f}" + f" {reference_param2_label}" + f"={param2}"
                    ax.scatter(zetas_reference, mse_over_zeta, s=16) 
                    ax.plot(zetas_reference, mse_over_zeta, "--", label=label) 

            if (reference_param1_legend_aggregated) and (reference_param1_legend_aggregated_select):
                #print(f"Values_per_level_index: {values_per_level_index}")
                rep_count_ref = len(values_per_level_index)
                values_per_index_np = np.array(values_per_level_index)
                values_per_index_np = values_per_index_np[values_per_index_np != 0.0]

                values[level_index] = np.mean(values_per_index_np)
                values_lp[level_index] = values[level_index] - np.std(values_per_index_np)
                values_up[level_index] = values[level_index] + np.std(values_per_index_np)
                
        #print(values)
        if (reference_param1_legend_aggregated):
            #label = r"reference. Tr. (varying $\xi_{tr}$'s)"
            label = reference_label
            print(f"Reference has {rep_count_ref} repetitions.")
            ax.scatter(zetas_reference[:], values[:], color="blue")
            ax.plot(zetas_reference[:], values[:], '-', label=label, color="blue")
            ax.fill_between(zetas_reference[:], values_lp[:], values_up[:], alpha=0.2, color="blue")

    # 1) load jittering  -> extract only specific points
    if (std_show):
        df_full_path = os.path.join(eval_df_save_dir_std, df_std_filename)
        if not os.path.exists(df_full_path):
            print(f"The file {df_full_path} does not exist.")
            return
        df = pd.read_csv(os.path.join(eval_df_save_dir_std, df_std_filename))

        tsm = {"train_noise_level" : "sigma_tr", "SGD_weight_decay" : "SGD_weight_decay",
             "zetas_start" : "zetas_start"}

        add_settings_info_to_df(df, t_set_mapping=tsm)
        df_red = df#.drop(["train_eps_constant"], axis=1)
        print(df)

        df_red.sort_values(by=std_param1_key, inplace=True)
        print(df_red)

        std_param1_key = "zetas_start"
        df_red.sort_values(by=std_param1_key, inplace=True)
    
        if (std_param1_restricted):
            param1_levels = (std_param1_selection)
        else:
            param1_levels = np.unique(df_red[std_param1_key])#.astype(np.float32).to_numpy()

        df_rows_for_param1_level = []
        for level in param1_levels:
            df_rows_for_param1_level.append(df_red[df_red[std_param1_key] == level])

        #print(df_rows_for_param1_level)

        values = np.zeros_like(zetas_std)
        values_up = np.zeros_like(zetas_std)
        values_lp = np.zeros_like(zetas_std)
        for level_index, level in enumerate(param1_levels):
            level = float(level)
            values_per_level_index = []
            for index, row in df_rows_for_param1_level[level_index].iterrows():
                param2 = row[std_param2_key]
                if (std_param2_restricted):
                    if param2 not in (std_param2_selection):
                        continue
                mse_over_zeta = row[zetas_std_columns].to_numpy()

                mse_over_zeta = mse_over_zeta.astype(np.float32)

                if((std_param1_legend_aggregated)):
                    if (std_param1_legend_aggregated_select):
                        values_per_level_index.append(mse_over_zeta[level_index])
                    else:
                        pass
                else:
                    label = f"{std_param1_label}" + f"={level:.2f}" + f" {std_param2_label}" + f"={param2}"
                    ax.scatter(zetas_std, mse_over_zeta, s=16) 
                    ax.plot(zetas_std, mse_over_zeta, "--", label=label) 

            if (std_param1_legend_aggregated) and (std_param1_legend_aggregated_select):
                print(f"Values_per_level_index: {values_per_level_index}")
                rep_count_jittering = len(values_per_level_index)
                values_per_index_np = np.array(values_per_level_index)
                values[level_index] = np.median(values_per_index_np)
                values_lp[level_index] = np.percentile(values_per_index_np, (100-perc)/2)
                values_up[level_index] = np.percentile(values_per_index_np, perc + (100-perc)/2)
                
        #print(values)
        if (std_param1_legend_aggregated):
            #label = r"Std. Tr. (varying $\xi_{tr}$'s)"
            label = std_label
            print(f"Jittering {rep_count_jittering} repetitions.")
            ax.scatter(zetas_std[:], values[:], color="orange")
            ax.plot(zetas_std[:], values[:], '-', label=label, color="orange")
            ax.fill_between(zetas_std[:], values_lp[:], values_up[:], alpha=0.2, color="orange")

    # 2) load standard trained models l2
    if (std_l2_show):
        df_full_path = os.path.join(eval_df_save_dir_stdl2, df_std_l2_filename)
        if not os.path.exists(df_full_path):
            print(f"The file {df_full_path} does not exist.")
            return
        df = pd.read_csv(os.path.join(eval_df_save_dir_stdl2, df_std_l2_filename))

        dsm = {"n" : "n"}
        tsm = {"train_noise_level" : "sigma_tr", "SGD_weight_decay" : "SGD_weight_decay",
             "zetas_start" : "zetas_start"}

        add_settings_info_to_df(df, t_set_mapping=tsm)
        #df["sigma_tr"] = df["train_eps_constant"].astype(np.float32) / df["n"].astype(np.float32).pow(0.5)
        df_red = df#.drop(["train_eps_constant"], axis=1)

        df_red.sort_values(by=std_param1l2_key, inplace=True)
        #print(df_red)
    
        if (std_param1l2_restricted):
            param1l2_levels = (std_param1l2_selection)
        else:
            param1l2_levels = np.unique(df_red[std_param1l2_key])#.astype(np.float32).to_numpy()

        print(f"param1l2_levels: {param1l2_levels}")

        # aggregate those dfs which are near each other
        df_rows_for_param1l2_level = []
        param1l2_levels_new = []
        i = 0
        while i < len(param1l2_levels):
            param1l2_levels_new.append(param1l2_levels[i])
            df_i = df_red[df_red[std_param1l2_key] == str(param1l2_levels[i])]
            j = i+1
            while j < len(param1l2_levels) and abs(float(param1l2_levels[i]) - float(param1l2_levels[j])) < 1e-6:
                df_i = pd.concat([df_i, df_red[df_red[std_param1l2_key] == str(param1l2_levels[j])]])
                i += 1; j+=1
            df_rows_for_param1l2_level.append(df_i)
            i+=1
        param1l2_levels = param1l2_levels_new


        values = np.zeros_like(zetas_stdl2)
        values_up = np.zeros_like(zetas_stdl2)
        values_lp = np.zeros_like(zetas_stdl2)
        for level_index, level in enumerate(param1l2_levels):
            level = float(level)
            if level_index < len(zetas_stdl2):
                values_per_level_index = []
                for index, row in df_rows_for_param1l2_level[level_index].iterrows():
                    param2l2 = row[std_param2l2_key]
                    run_dir = row["run_dir"]

                    if (std_param2l2_restricted):
                        if param2l2 not in (std_param2l2_selection):
                            continue
                    mse_over_zeta = row[zetas_stdl2_columns].to_numpy()

                    mse_over_zeta = mse_over_zeta.astype(np.float32)

                    if((std_param1l2_legend_aggregated)):
                        if (std_param1l2_legend_aggregated_select):
                            values[level_index] = mse_over_zeta[level_index]
                            values_per_level_index.append(mse_over_zeta[level_index])
                        else:
                            values[level_index] = np.median(mse_over_zeta)
                            values_lp[level_index] = np.percentile(mse_over_zeta, (100-perc)/2)
                            values_up[level_index] = np.percentile(mse_over_zeta, perc + (100-perc)/2)
                    else:
                        label = f"{std_param1l2_label}" + f"={level:.2f}" + f" {std_param2l2_label}" + f"={param2l2}"
                        ax.scatter(zetas_stdl2, mse_over_zeta, s=16, label=label) 
                        ax.plot(zetas_stdl2, mse_over_zeta, "--", label=label) 
                if (std_param1_legend_aggregated) and (std_param1_legend_aggregated_select):
                    #print(f"Values_per_level_index: {values_per_level_index}")
                    rep_count_l2 = len(values_per_level_index)
                    values_per_index_np = np.array(values_per_level_index)
                    values[level_index] = np.median(values_per_index_np)
                    values_lp[level_index] = np.percentile(values_per_index_np, (100-perc)/2)
                    values_up[level_index] = np.percentile(values_per_index_np, perc + (100-perc)/2)
            else:
                print(f"Skip level index: {level_index}")
        #print(values)
        if (std_param1l2_legend_aggregated):
            #label = r"Std. Tr. (varying $\xi_{tr}$'s)"
            label = std_l2_label
            print(f"Weight decay: {rep_count_l2} repetitions.")
            ax.scatter(zetas_stdl2, values, color="brown")
            ax.plot(zetas_stdl2, values, '-', label=label, color="brown")
            ax.fill_between(zetas_stdl2, values_lp, values_up, alpha=0.2, color="brown")

    # 2) load standard trained models errReg
    if (std_err_reg_show):
        df_full_path = os.path.join(eval_df_save_dir_stdErrorReg, df_std_err_reg_filename)
        if not os.path.exists(df_full_path):
            print(f"The file {df_full_path} does not exist.")
            return
        df = pd.read_csv(os.path.join(eval_df_save_dir_stdErrorReg, df_std_err_reg_filename))

        dsm = {"n" : "n"}
        tsm = {"train_noise_level" : "sigma_tr", "SGD_weight_decay" : "SGD_weight_decay",
             "zetas_start" : "zetas_start"}

        add_settings_info_to_df(df, t_set_mapping=tsm)
        df_red = df#.drop(["train_eps_constant"], axis=1)

        df_red.sort_values(by=std_param1_err_reg_key, inplace=True)

        if (std_param1_err_reg_restricted):
            param1_err_reg_levels = (std_param1_err_reg_selection)
        else:
            param1_err_reg_levels = np.unique(df_red[std_param1_err_reg_key])#.astype(np.float32).to_numpy()

        df_rows_for_param1_err_reg_level = []
        for level in param1_err_reg_levels:
            df_rows_for_param1_err_reg_level.append(df_red[df_red[std_param1_err_reg_key] == str(level)])

        values = np.zeros_like(zetas_stdErrReg)
        values_up = np.zeros_like(zetas_stdErrReg)
        values_lp = np.zeros_like(zetas_stdErrReg)
        for level_index, level in enumerate(param1_err_reg_levels):
            level = float(level)
            values_per_level_index = []
            if level_index < len(zetas_stdErrReg):
                for index, row in df_rows_for_param1_err_reg_level[level_index].iterrows():
                    param2errReg = row[std_param2_err_reg_key]
                    run_dir = row["run_dir"]

                    if (std_param2_err_reg_restricted):
                        if param2errReg not in (std_param2_err_reg_selection):
                            continue
                    mse_over_zeta = row[zetas_std_err_reg_columns].to_numpy()


                    mse_over_zeta = mse_over_zeta.astype(np.float32)

                    if((std_param1_err_reg_legend_aggregated)):
                        if (std_param1_err_reg_legend_aggregated_select):
                            #values[level_index] = mse_over_zeta[level_index]
                            values_per_level_index.append(mse_over_zeta[level_index])
                        else:
                            values[level_index] = np.median(mse_over_zeta)
                            values_lp[level_index] = np.percentile(mse_over_zeta, (96-perc)/2)
                            values_up[level_index] = np.percentile(mse_over_zeta, perc + (96-perc)/2)
                    else:
                        label = f"{std_param1_err_reg_label}" + f"={level:.2f}" + f" {std_param2_err_reg_label}" + f"={param2errReg}"
                        ax.scatter(zetas_stdErrReg, mse_over_zeta, s=16, label=label) 
                        ax.plot(zetas_stdErrReg, mse_over_zeta, "--", label=label) 
            if (std_param1_legend_aggregated) and (std_param1_legend_aggregated_select):
                #print(f"Values_per_level_index: {values_per_level_index}")
                rep_count_errReg = len(values_per_level_index)
                values_per_index_np = np.array(values_per_level_index)
                values[level_index] = np.median(values_per_index_np)
                values_lp[level_index] = np.percentile(values_per_index_np, (100-perc)/2)
                values_up[level_index] = np.percentile(values_per_index_np, perc + (100-perc)/2)
            else:
                print(f"Skip level index: {level_index}")
        #print(values)
        if (std_param1_err_reg_legend_aggregated):
            #label = r"Std. Tr. (varying $\xi_{tr}$'s)"
            label = std_err_reg_label
            print(f"Error Regularization: {rep_count_errReg} repetitions.")
            ax.scatter(zetas_stdErrReg[:], values[:], color="purple")
            ax.plot(zetas_stdErrReg[:], values[:], '-', label=label, color="purple")
            ax.fill_between(zetas_stdErrReg[:], values_lp[:], values_up[:], alpha=0.2, color="purple")

    # 2) load adversarial baseline models
    if (adv_show):
        df_full_path = os.path.join(eval_df_save_dir_adv, df_adv_filename)
        if not os.path.exists(df_full_path):
            print(f"The file {df_full_path} does not exist.")
            return
        df = pd.read_csv(os.path.join(eval_df_save_dir_adv, df_adv_filename))

        #print(f"adv_path: {os.path.join(df_dir_path, df_adv_filename)})")

        if not (overwrite_global):
            plot_figsize = same_sigma_figsize = global_figsize
            plot_show = same_sigma_show = global_plots_show

        tsm = {"train_noise_level" : "sigma_tr", "SGD_weight_decay" : "SGD_weight_decay",
             "zetas_start" : "zetas_start"}

        add_settings_info_to_df(df, t_set_mapping=tsm)
        df_red = df#.drop(["train_eps_constant"], axis=1)

        adv_param1_key = "zetas_start"

        if (adv_param1_restricted):
            param1_levels = (adv_param1_selection)
        else:
            param1_levels = np.unique(df_red[adv_param1_key])#.astype(np.float32).to_numpy()

        print(param1_levels)

        df_rows_for_param1_level = []
        for level in param1_levels:
            df_rows_for_param1_level.append(df_red[df_red[adv_param1_key] == str(level)])

        values = np.zeros_like(zetas_adv)
        values_up = np.zeros_like(zetas_adv)
        values_lp = np.zeros_like(zetas_adv)
        for level_index, level in enumerate(param1_levels):
            series = []
            for index, row in df_rows_for_param1_level[level_index].iterrows():
                param2 = row[adv_param2_key]
                mse_over_zeta = row[zetas_adv_columns].to_numpy()

                mse_over_zeta = mse_over_zeta.astype(np.float32)

                if((adv_param2_legend_hide)):
                    label = f"Rob. Tr. ({adv_param1_label}" + f"={float(level):.1f})"
                else:
                    label = f"Rob. Tr. ({adv_param1_label}" + f"={float(level):.2f}" + f" {adv_param2_label}" + f"={param2}"

                if not adv_aggregate:
                    ax.scatter(zetas_adv, mse_over_zeta, s=16)
                    ax.plot(zetas_adv, mse_over_zeta, "-", label=label) 
                else:
                    if adv_aggregate_select:
                        series.append(mse_over_zeta[level_index])
                    else:
                        series.append(mse_over_zeta)

            if adv_aggregate:
                series_np = np.array(series)
                print(f"Adversarial Training: {series_np.shape[0]} repetitions.")

                if adv_aggregate_select:
                    values[level_index] = np.median(series_np)
                    values_lp[level_index] = np.percentile(series_np, (100-perc)/2)
                    values_up[level_index] = np.percentile(series_np, perc + (100-perc)/2)
                else:
                    values = np.mean(series_np, axis=(0))
                    values_lp = values - np.std(series_np, axis=(0))
                    values_up = values + np.std(series_np, axis=(0))
        if adv_aggregate:
            label = f"Robust Training"
            ax.scatter(zetas_adv[:], values[:], color="black")
            ax.plot(zetas_adv[:], values[:], label=label, color="black")
            ax.fill_between(zetas_adv[:], values_lp[:], values_up[:], alpha=0.2, color="black")

    #ax.legend(loc="lower right")
    #ax.set_ylim([0, 1.4])
    #ax.set_xlim([0, 1.0])
    handles, labels = ax.get_legend_handles_labels()
    if not (plot_create_tikz):
        ax.legend(handles[::-1], labels[::-1], loc='lower right')
    #ax.set_ylim([0, 0.8])
    #fig.tight_layout()
    plot_filename = os.path.join(base_output_dir, plot_filename)
    print(base_output_dir)
    _save_create_tikz_fig(fig, plot_save, plot_create_tikz, plot_filename, global_dpi)
    if (plot_show):
        fig.show()
    else:
        plt.close(fig)