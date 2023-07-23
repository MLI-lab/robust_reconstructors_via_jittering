from experiments.subspace.configs import (
    train_configs,
    hpsearch_configs,
    eval_configs,
    plot_configs
)

step_config_dict = {
    "hpsearch"       : hpsearch_configs.step_configs,
    "train"          : train_configs.train_step_configs,
    "eval"           : eval_configs.eval_step_configs,
    "plot"           : plot_configs.plot_step_configs
}