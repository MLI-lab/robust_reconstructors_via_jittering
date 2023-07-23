from typing import Callable
import os
import shutil
from configparser import ConfigParser
from datetime import datetime
from tensorboard import program
import time

temp_file_name = "started_run.txt"
def _create_temp_file(log_dir):
    sr_path = os.path.join(log_dir, temp_file_name)
    if os.path.exists(sr_path):
        print(f"sr_path already exists: {sr_path}")
        return True
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        f = open(sr_path, "x")
        f.write("Is running..")
        f.close()
        return False

def _has_temp_file(log_dir):
    return os.path.exists(os.path.join(log_dir, temp_file_name))

def _logdir_exists(log_dir):
    return os.path.exists(log_dir)

def _has_empty_logdir(log_dir):
    return os.path.isdir(log_dir) and len(os.listdir(log_dir)) == 0

class ExperimentStep():
    step_func : Callable
    artifact_path : str
    parameters : dict

    def __init__(self, step_func : Callable, artifact_path : str, parameters : dict):
        self.step_func = step_func
        self.artifact_path = artifact_path
        self.parameters = parameters

    def describe(self, indentation):
        print(f"{indentation}Artifact_path: {self.artifact_path}.")

    def get_status(self):
        if (not _logdir_exists(self.artifact_path)) or _has_empty_logdir(self.artifact_path):
            return "clear"
        elif _has_temp_file(self.artifact_path):
            return "started"
        else:
            return "finished"

    def clear_artifacts(self, only_started=True):
        if not only_started or _has_temp_file(self.artifact_path):
            if os.path.exists(self.artifact_path):
                shutil.rmtree(self.artifact_path)

    def tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", self.artifact_path])
        url = tb.launch()
        print(f"Tensorboard listening on {url}")
        while True:
            time.sleep(60)

    def show_artifacts(self, indentation):
        if not _logdir_exists(self.artifact_path):
            print(f"{indentation}Logdir does not exist.")
        else:
            print(f"{indentation}Artifacts:")
            for item in os.listdir(self.artifact_path):
                print(f"{indentation} {item}")

    def _save_parameters(self):
        config = ConfigParser(allow_no_value=True)
        config.add_section("Parameters")
        print(self.parameters)
        config["Parameters"] = self.parameters
        config.add_section("Step")
        config["Step"] = {
            "timestamp_saved" : datetime.now()
        }
        with open(os.path.join(self.artifact_path, "settings"), "w") as configfile:
            config.write(configfile)

    def run(self, devices):
        print(f"Running experiment step..")
        _create_temp_file(self.artifact_path)
        self._save_parameters()
        self.step_func(**self.parameters, devices=devices)
        os.remove(os.path.join(self.artifact_path, temp_file_name))