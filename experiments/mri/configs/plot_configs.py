

import os

from experiments.mri.configs.base_config import (
    base_artifact_path__default
)
step_name = "plot"

global_config = {
    "global_figsize" : (8,6),
    "global_create_tikz" : False,
    "global_dpi" : 100,
    "global_plots_show" : False
}

unet_base_plot_config = {
    "overwrite_global" :  False,
    "plot_figsize" :  (8,6),
    "plot_show" :  False,
    "plot_save" :  True,
    #"plot_filename" :  "pub/unet_standard_comparision",
    "plot_custom_colors" :  True,
    "plot_selected_colors" :  ['purple', 'orange', 'red', 'green', "blue", "brown", "black", "gray", "lightblue", "darkblue"],
    "plot_linestyles" :  ['-', '--', ":", "-", "*", "-", "-", "-"],
    "reference_show" :  True,
    "reference_param2_restricted" :  False,
    "reference_param2_selection" :  [0.5],
    "reference_param1_restricted" :  False,
    "reference_param1_selection" :  [0.5, 0.8470085263252258],
    "reference_param1_colors" :  True,
    "reference_param1_fixed_color" :  True,
    "reference_param1_fixed_color_selection" :  'red',
    "reference_param1_legend_aggregated" :  True,
    "reference_param1_legend_aggregated_select" :  True,
    "reference_param1_key" :  "sigma_tr",
    "reference_param2_key" :  "sigma_te",
    "reference_param1_label" :  "$\sigma_{tr}$",
    "reference_param2_label" :  "$r$",
    "reference_label" :  "Reference",
    "std_show" :  True,
    "std_param2_restricted" :  False,
    "std_param2_selection" :  [0.5],
    "std_param1_restricted" :  False,
    "std_param1_selection" :  [0.5, 0.8470085263252258],
    "std_param1_colors" :  True,
    "std_param1_fixed_color" :  True,
    "std_param1_fixed_color_selection" :  'red',
    "std_param1_legend_aggregated" :  True,
    "std_param1_legend_aggregated_select" :  True,
    "std_param1_key" :  "sigma_tr",
    "std_param2_key" :  "sigma_te",
    "std_param1_label" :  "$\sigma_{tr}$",
    "std_param2_label" :  "$r$",
    "std_label" :  "Jittering",
    "std_l2_show" :  False,
    "std_param2l2_restricted" :  False,
    "std_param2l2_selection" :  [0.5],
    "std_param1l2_restricted" :  False,
    "std_param1l2_selection" :  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "std_param1l2_colors" :  True,
    "std_param1l2_fixed_color" :  True,
    "std_param1l2_fixed_color_selection" :  'red',
    "std_param1l2_legend_aggregated" :  False,
    "std_param1l2_legend_aggregated_select" :  False,
    "std_param1l2_key" :  "SGD_weight_decay",
    "std_param2l2_key" :  "sigma_te",
    "std_param1l2_label" :  "$\sigma_{tr}$",
    "std_param2l2_label" :  "$r$",
    "std_err_reg_show" :  False,
    "std_param2_err_reg_restricted" :  False,
    "std_param2_err_reg_selection" :  [0.5],
    "std_param1_err_reg_restricted" :  False,
    "std_param1_err_reg_selection" :  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "std_param1_err_reg_colors" :  True,
    "std_param1_err_reg_fixed_color" :  True,
    "std_param1_err_reg_fixed_color_selection" :  'red',
    "std_param1_err_reg_legend_aggregated" :  True,
    "std_param1_err_reg_legend_aggregated_select" :  True,
    "std_param1_err_reg_key" :  "ErrorReg_param",
    "std_param2_err_reg_key" :  "sigma_te",
    "std_param1_err_reg_label" :  "$\sigma_{tr}$",
    "std_param2_err_reg_label" :  "$r$",
    "std_err_reg_label" :  "Jacobian",
    "std_l2_label" :  "Weight Decay",
    "adv_show" :  True,
    "adv_aggregate" :  True,
    "adv_aggregate_select" : False,
    "adv_param2_restricted" :  False,
    "adv_param2_selection" :  [0.19999999],
    "adv_param1_restricted" :  False,
    "adv_param1_selection" :  [0.19999999],
    "adv_param1_colors" :  True,
    "adv_param1_key" :  "sigma_tr",
    "adv_param2_key" :  "sigma_te",
    "adv_param1_label" :  r"$\xi_{tr}$",
    "adv_param2_label" :  r"$r$",
    "adv_param2_legend_hide" :  True,
    "plot_ylabel" : "R"
}


from experiments.mri.configs.eval_configs import (
    evaluate_UNet_adversarial_mri,
    evaluate_UNet_jittering_mri,
    evaluate_UNet_standard_mri,
)

unet_mri = {
    "base_dir_reference"            : evaluate_UNet_standard_mri["base_dir"],
    "base_dir_std"                  : evaluate_UNet_jittering_mri["base_dir"],
    "base_dir_stdl2"                : None,
    "base_dir_stdErrorReg"          : None,
    "base_dir_adv"                  : evaluate_UNet_adversarial_mri["base_dir"],
    "eval_df_save_dir_reference"    : evaluate_UNet_standard_mri["df_save_dir"],
    "eval_df_save_dir_std"          : evaluate_UNet_jittering_mri["df_save_dir"],
    "eval_df_save_dir_stdl2"        : None,
    "eval_df_save_dir_stdErrorReg"  : None,
    "eval_df_save_dir_adv"          : evaluate_UNet_adversarial_mri["df_save_dir"],
    "noise_levels_stdRef"           : evaluate_UNet_standard_mri["fixed_noise_levels"],
    "noise_levels_std"              : evaluate_UNet_jittering_mri["fixed_noise_levels"],
    "noise_levels_stdl2"            : None,
    "noise_levels_stdErrorReg"      : None,
    "noise_levels_adv"              : evaluate_UNet_adversarial_mri["fixed_noise_levels"],
    "global_plot_config"            : global_config,
    "specific_plot_config"          : unet_base_plot_config,
    "base_output_dir"               : os.path.join(base_artifact_path__default, step_name, "unet_mri"),
    "plot_filename_std"             : "unet_standard_comparison",
    "plot_filename_adv"             : "unet_adversarial_comparison"
}


from generic.step import ExperimentStep
from experiments.mri.steps.plot import plot

plot_step_configs = {
    "unet_mri" : ExperimentStep(
        step_func=plot,
        artifact_path=unet_mri["base_output_dir"],
        parameters=unet_mri
    )
}