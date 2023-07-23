from experiments.natural_images.configs import (
    preprocess_configs,
    train_configs,
    hpsearch_configs,
    eval_configs,
    plot_configs,
    visualize_configs
)

step_config_dict = {
    "preprocess"     : preprocess_configs.step_configs,
    "hpsearch"       : hpsearch_configs.step_configs,
    "train"          : train_configs.train_step_configs,
    "eval"           : eval_configs.eval_step_configs,
    "plot"           : plot_configs.plot_step_configs,
    "visualize"      : visualize_configs.step_configs,
    "convergence"    : None,
    "data_analysis"  : None
}