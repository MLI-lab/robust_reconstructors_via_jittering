# %%
import torch
import os
import pandas as pd
import subprocess
from io import StringIO
import time

def get_last_gpu__or__cpu():
    prefer_cpu = False
    if prefer_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device = "cuda:" + str(device_count-1)

    if device == "cpu":
        torch.set_num_threads(16)
        torch.set_num_interop_threads(16)

    print(f"Using {device} device")
    return device

def get_device_id(ind):
    if torch.cuda.device_count() == 1:
        return "cuda"
    else:
        return f"cuda:{ind}"

def get_all_devices():
    c = torch.cuda.device_count()
    if c == 1:
        return "cuda"
    else:
        return [f"cuda:{i}" for i in range(c)]


def get_free_devices_nvidia_smi(min_gpu : float = 80, min_mem : float = 4000):
    df = get_device_infos_nvidia_smi()
    df["gpu_util"] = df[" utilization.gpu [%]"].map(lambda x: float(x.replace("%", "")))
    df["mem_util"] = df[" utilization.memory [%]"].map(lambda x: float(x.replace("%", "")))
    df["mem_free"] = df[" memory.free [MiB]"].map(lambda x: float(x.replace("MiB", "")))
    df_filtered = df[ (df["gpu_util"] < (100-min_gpu)) & (df["mem_free"] > min_mem)]
    indices = list(df_filtered["index"])
    if len(indices) == 0:
        return ["cpu"]
    else:
        return [get_device_id(i) for i in indices]

def get_free_devices_nvidia_smi_rep(min_gpu : float = 80, min_mem : float = 4000, rep : int = 2, wait_time : float = 0.1):
    devices = get_free_devices_nvidia_smi(min_gpu=min_gpu, min_mem=min_mem)
    if len(devices) == 1 and devices[0] == "cpu":
        return devices
    for _ in range(rep-1):
        time.sleep(wait_time)
        new_devices = get_free_devices_nvidia_smi(min_gpu=min_gpu, min_mem=min_mem)
        devices = [device for device in devices if device in new_devices]

    if len(devices) == 0:
        return ["cpu"]
    
    return devices


def print_device_infos_cuda():
    device_nrs = torch.cuda.device_count()
    for d_ind in range(device_nrs):
        device = torch.device(d_ind)#"cuda" if device_nrs == 1 else f"cuda:{d_ind}")
        if device.type == 'cuda':
            print(f"---- device: {get_device_id(d_ind)} ------")
            print(f"Device: {torch.cuda.get_device_name(d_ind)}")
            print('Memory Usage:')
            print('Allocated:', torch.cuda.memory_allocated(d_ind), 'GB')
            print('Cached:   ', torch.cuda.memory_reserved(d_ind), 'GB')
        
def get_device_infos_nvidia_smi():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory", "--format=csv"], stdout=subprocess.PIPE)
    txt = result.stdout.decode("utf-8")
    return pd.read_csv(StringIO(txt))

def print_device_infos_nvidia_smi():
    print(get_device_infos_nvidia_smi())

def print_device_infos_cuda():
    device_nrs = torch.cuda.device_count()
    for d_ind in range(device_nrs):
        device = torch.device(d_ind)#"cuda" if device_nrs == 1 else f"cuda:{d_ind}")
        if device.type == 'cuda':
            print(f"---- device: {get_device_id(d_ind)} ------")
            print(f"Device: {torch.cuda.get_device_name(d_ind)}")
            print('Memory Usage:')
            print('Allocated:', torch.cuda.memory_allocated(d_ind), 'GB')
            print('Cached:   ', torch.cuda.memory_reserved(d_ind), 'GB')