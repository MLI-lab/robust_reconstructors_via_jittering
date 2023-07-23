import argparse
import re

import experiments.natural_images.step_config_mapping as nat_img_steps
import experiments.mri.step_config_mapping as mri_steps
import experiments.subspace.step_config_mapping as subspace_steps
from typing import Callable

from generic.helper.device_helpers import get_free_devices_nvidia_smi_rep

exp_name_mri = "mri"
exp_name_nat_images = "nat_img"
exp_name_subspace_model = "subspace"

exp_step_dict = {
    exp_name_nat_images : nat_img_steps.step_config_dict,
    exp_name_mri : mri_steps.step_config_dict,
    exp_name_subspace_model : subspace_steps.step_config_dict
}

def _reg_filter_on_list(reg_filter : str, str_list : list[str]):
    p = re.compile(reg_filter) 
    return list(filter(lambda x : p.match(x) is not None, str_list))

def _for_each_step_config(experiments : str, steps : str, configs : str, funct : Callable, verbose : bool = False, stop_first : bool = False):
    matching_exps = _reg_filter_on_list(experiments, list(exp_step_dict.keys()))
    for exp in matching_exps:
        if verbose:
            print(f"Experiment: {exp}")
        step_dict = exp_step_dict[exp]
        if step_dict is not None:
            matching_steps = _reg_filter_on_list(steps, list(exp_step_dict[exp].keys()))
            for step_ind, step in enumerate(matching_steps):
                prepend_label_step = "└───" if step_ind == len(matching_steps)-1 else "├───"
                prepend_label_conf = "    " if step_ind == len(matching_steps)-1 else "│   "
                step_config_dict = step_dict[step]
                if step_config_dict is not None:
                    matching_configs = _reg_filter_on_list(configs, list(step_config_dict.keys()))
                    if len(matching_configs) > 0 and verbose:
                        print(f" {prepend_label_step}Step: {step}")
                    for exp_step_ind, exp_step in enumerate(matching_configs):
                        prepend_label_exp_step = "└───" if exp_step_ind == len(matching_configs)-1 else "├───"
                        prepend_label_exp_step_output = "    " if exp_step_ind == len(matching_configs)-1 else "│   "
                        if verbose:
                            print(f" {prepend_label_conf}{prepend_label_exp_step}Conf: {exp_step} ({step_config_dict[exp_step].get_status()})")
                        funct(step_config_dict[exp_step], (prepend_label_conf, prepend_label_exp_step_output))
                        if stop_first:
                            return

def run_steps(experiments, steps, configs):
    devices = get_free_devices_nvidia_smi_rep(min_gpu=80, min_mem=10000, rep=2)
    print(f"Found free devices: {devices}")
    print(f"Run steps with: {experiments} and steps {steps} and configs {configs}.")
    _for_each_step_config(experiments, steps, configs, lambda x,_: x.run(devices))

def clear_steps(experiments, steps, configs):
    print(f"Clear steps with: {experiments} and steps {steps} and configs {configs}.")
    _for_each_step_config(experiments, steps, configs, lambda x,_: x.clear_artifacts(only_started=True))

def remove_steps(experiments, steps, configs):
    print(f"Remove steps with: {experiments} and steps {steps} and configs {configs}.")
    _for_each_step_config(experiments, steps, configs, lambda x,_: x.clear_artifacts(only_started=False))

def list_steps(experiments, steps, configs):
    print(f"List steps with: {experiments} and steps {steps} and configs {configs}.")
    _for_each_step_config(experiments, steps, configs, lambda x,_: None, verbose=True)

def show_steps(experiments, steps, configs):
    print(f"Show steps with: {experiments} and steps {steps} and configs {configs}.")
    def show_step(x,lbls):
        lbl1,lbl2 = lbls
        x.describe(indentation=f" {lbl1}{lbl2}  ")
        x.show_artifacts(indentation=f" {lbl1}{lbl2}  ")
    _for_each_step_config(experiments, steps, configs, lambda x,lbls: show_step(x,lbls), verbose=True)

def tb_steps(experiments, steps, configs):
    print(f"Tb steps with: {experiments} and steps {steps} and configs {configs}.")
    _for_each_step_config(experiments, steps, configs, lambda x,_: x.tensorboard(), verbose=False, stop_first=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Code for 'Learning Provably Robust Estimators for Inverse Problems via Jittering'",
        description='The program performs the experimental pipelines to reproduce results of the paper.',
        epilog='')

    parser.add_argument('cmd', choices=['run', 'list', 'clear', 'remove', 'tb', 'show'],
                        help='Command (run, clear, tb).')
    parser.add_argument('-e', '--exps', choices=['mri', 'nat_img', 'subspace'], default='.*',
                        help='Type of experiment (mri, nat_img, subspace).')
    parser.add_argument('-s', '--steps', type=str, default='.*',
                        help='Steps to perform (preprocess, hpsearch, train, eval, plot, visualize).')
    parser.add_argument('-c', '--configs', type=str, default='.*',
                        help='Step configurations to consider (differs for each step and exp).')

    args = parser.parse_args()
    cmd, experiments, steps, configs = args.cmd, args.exps, args.steps, args.configs

    cmd_map = {
        "run" : run_steps,
        "list" : list_steps,
        "clear" : clear_steps,
        "remove" : remove_steps,
        "show" : show_steps,
        "tb" : tb_steps,
    }
    cmd_map[cmd](experiments, steps, configs)
