step_name = "preprocess"

preprocessing_config_train = {
    "n" : 6400,
    "d" : 4000,
    "size"  : 50000,
    "noise_level" : 0.2,
    "linear_forward_name" : "None",
    "linear_forward_kwargs" : {}
}
preprocessing_config_val = {
    "n" : 6400,
    "d" : 4000,
    "size"  : 10000,
    "noise_level" : 0.2,
    "linear_forward_name" : "None",
    "linear_forward_kwargs" : {}
}
preprocessing_config_test = {
    "n" : 6400,
    "d" : 4000,
    "size"  : 100,
    "noise_level" : 0.2,
    "linear_forward_name" : "None",
    "linear_forward_kwargs" : {}
}

preprocessing_config_linear_decay_train = {
    "n" : 100,
    "d" : 80,
    "size"  : 50000,
    "noise_level" : 0.2,
    "linear_forward_name" : "linear_decay",
    "linear_forward_kwargs" : {"index_start" : 1, "index_end" : 100, "rep_per_value" : 1}
}
preprocessing_config_linear_decay_val = {
    "n" : 100,
    "d" : 80,
    "size"  : 10000,
    "noise_level" : 0.2,
    "linear_forward_name" : "linear_decay",
    "linear_forward_kwargs" : {"index_start" : 1, "index_end" : 100, "rep_per_value" : 1}
}
preprocessing_config_linear_decay_test = {
    "n" : 100,
    "d" : 80,
    "size"  : 10000,
    "noise_level" : 0.2,
    "linear_forward_name" : "linear_decay",
    "linear_forward_kwargs" : {"index_start" : 1, "index_end" : 100, "rep_per_value" : 1}
}