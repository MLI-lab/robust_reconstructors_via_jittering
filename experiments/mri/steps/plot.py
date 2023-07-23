from generic.generate_plots import gen_jittering_std_adv_comparision_aggregatedl2errReg
import os

def plot(
        base_dir_reference, base_dir_std, base_dir_stdl2, base_dir_stdErrorReg, base_dir_adv,
        eval_df_save_dir_reference, eval_df_save_dir_std, eval_df_save_dir_stdl2, eval_df_save_dir_stdErrorReg, eval_df_save_dir_adv,
        noise_levels_stdRef, noise_levels_std, noise_levels_stdl2, noise_levels_stdErrorReg, noise_levels_adv,
        base_output_dir,
        plot_filename_std, plot_filename_adv,
        global_plot_config, specific_plot_config,
        df_reference_filename_std = "std.csv", df_std_filename_std = "std.csv", df_std_l2_filename_std = "std.csv", df_std_err_reg_filename_std = "std.csv", df_adv_filename_std = "std.csv",
        df_reference_filename_adv = "adv.csv", df_std_filename_adv = "adv.csv", df_std_l2_filename_adv = "adv.csv", df_std_err_reg_filename_adv = "adv.csv", df_adv_filename_adv = "adv.csv",
        devices = ["cpu"]):

    if base_dir_reference is not None:
        base_dir_reference = os.path.join(base_dir_reference, "*", "*")
    if base_dir_std is not None:
        base_dir_std = os.path.join(base_dir_std, "*", "*")
    if base_dir_stdl2 is not None:
        base_dir_stdl2 = os.path.join(base_dir_stdl2, "*", "*")
    if base_dir_stdErrorReg is not None:
        base_dir_stdErrorReg = os.path.join(base_dir_stdErrorReg, "*", "*")
    if base_dir_adv is not None:
        base_dir_adv = os.path.join(base_dir_adv, "*", "*")
    

    # standard risk
    gen_jittering_std_adv_comparision_aggregatedl2errReg(
        base_dir_reference, base_dir_std, base_dir_stdl2, base_dir_stdErrorReg, base_dir_adv,
        eval_df_save_dir_reference, eval_df_save_dir_std, eval_df_save_dir_stdl2, eval_df_save_dir_stdErrorReg, eval_df_save_dir_adv,
        noise_levels_stdRef, noise_levels_std, noise_levels_stdl2, noise_levels_stdErrorReg, noise_levels_adv,
        df_reference_filename_std, df_std_filename_std, df_std_l2_filename_std, df_std_err_reg_filename_std, df_adv_filename_std,
        plot_filename=plot_filename_std,base_output_dir=base_output_dir,
        **global_plot_config, **specific_plot_config)

    # robust risk
    gen_jittering_std_adv_comparision_aggregatedl2errReg(
        base_dir_reference, base_dir_std, base_dir_stdl2, base_dir_stdErrorReg, base_dir_adv,
        eval_df_save_dir_reference, eval_df_save_dir_std, eval_df_save_dir_stdl2, eval_df_save_dir_stdErrorReg, eval_df_save_dir_adv,
        noise_levels_stdRef, noise_levels_std, noise_levels_stdl2, noise_levels_stdErrorReg, noise_levels_adv,
        df_reference_filename_adv, df_std_filename_adv, df_std_l2_filename_adv, df_std_err_reg_filename_adv, df_adv_filename_adv,
        plot_filename=plot_filename_adv,base_output_dir=base_output_dir,
        **global_plot_config, **specific_plot_config)