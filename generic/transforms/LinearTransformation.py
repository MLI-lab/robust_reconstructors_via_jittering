import torch
import torch.nn as nn
import math
import numpy as np

def create_diagonal_by_name(name, size, kwargs_dict):
    if name == "None":
        return torch.ones(size)
    elif name == "linear_decay":
        index_start = kwargs_dict["index_start"]; index_end = kwargs_dict["index_end"]; rep_per_value = kwargs_dict["rep_per_value"]
        return torch.tensor([i / index_end for i in range(index_start, index_end+1)]*rep_per_value)
    elif name == "geom_decay":
        base = kwargs_dict["base"]; index_start = kwargs_dict["index_start"]; index_end = kwargs_dict["index_end"]; rep_per_value = kwargs_dict["rep_per_value"]
        return torch.tensor([math.pow(base, i) for i in range(index_start, index_end+1)]*rep_per_value)
    elif name == "diag_two_sing_values":
        s1 = kwargs_dict["singv1"]; s2 = kwargs_dict["singv2"]
        ratio = kwargs_dict["ratio"]
        print(f"Use diag_two_sing_values with s1={s1}, s2={s2} and ratio={ratio}")
        size1 = int(float(ratio)*size)
        return    torch.concat(
                (torch.ones( size1 ) * s1, torch.ones(size-size1)*s2)
            )
    if name == "half_0.01_half_1.0":
        print("Use default diagonal form (half 0.001, half 1)")
        return    torch.concat(
                (torch.ones(size//2) * 0.001, torch.ones(size-size//2))
            )
    elif name == "half_0.0001_half_1.0":
        print("Use default diagonal form (half 0.0001, half 1)")
        return    torch.concat(
                (torch.ones(size//2) * 0.0001, torch.ones(size-size//2))
            )
    else:
        print("Use default diagonal form (half 0.1, half 1)")
        return    torch.concat(
                (torch.ones(size//2) * 0.1, torch.ones(size-size//2))
            )
    #)

class LinearTransformation(object):
    def __init__(self, matrix_data, is_diagonal, device="cpu"):
        self._is_diagonal = is_diagonal
        matrix_data = matrix_data.to(device)
        if is_diagonal:
            self._diagonal = nn.Parameter(matrix_data)
        else:
            n = matrix_data.shape[-1]
            self._linear = nn.Linear(n, n, bias=False)
            self._linear_weight = nn.Parameter(matrix_data)
            self._linear.weight.requires_grad_(False)
            self._linear = self.linear.float()
        
    def __call__(self, tensor):
        if self._is_diagonal:
            return tensor.float() * self._diagonal
        else:
            if len(tensor.shape) < len(self._linear.weight.data.shape):
                t = torch.unsqueeze(tensor, dim=0)
                return torch.squeeze(
                    self._linear(t)
                , dim=0)
            else:
                return self._conv_filter(t)
    
    def __repr__(self):
        return self.__class__.__name__