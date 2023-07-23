import torch.multiprocessing as mp
import numpy as np

def distr_configs_over_gpus2(devices, params, trainfunc, nr_per_device=1):
    pdevices = list(np.repeat(np.array(devices), nr_per_device))
    batch_count = len(params) // len(pdevices)
    rem_steps = len(params) % len(pdevices)
    print(f"Spawning on pdevices {pdevices}")
    for i in range(batch_count):
        mp.spawn(trainfunc, args=(i*len(pdevices), pdevices, params,), nprocs=len(pdevices), join=True, daemon=False, start_method='spawn')
    if rem_steps > 0:
        subpdevices = pdevices[-rem_steps:]
        mp.spawn(trainfunc, args=(batch_count*len(pdevices), subpdevices, params,), nprocs=len(subpdevices), join=True, daemon=False, start_method='spawn')