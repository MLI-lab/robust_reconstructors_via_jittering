# %%
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

def construct_UUt(n, d):
    U = np.identity(n)[:, 0:d]
    UUT = np.matmul(U, U.transpose())
    return torch.from_numpy(UUT)

class NeuralNetwork(nn.Module):
    def __init__(self, n, d, device, skip_linear=False, not_skip_linear_diagonal=False, full_linear=False):
        super(NeuralNetwork, self).__init__()
        if full_linear:
            self.linear = nn.Linear(n, n, bias=False).float()
        elif not skip_linear:
            if not_skip_linear_diagonal:
                self.linear_diagonal = nn.Parameter(torch.rand(n))
                self.linear_diagonal.requires_grad = True
            else:
                self.linear = nn.Linear(n, n, bias=False)
                self.linear.weight = nn.Parameter(construct_UUt(n, d))
                self.linear.weight.requires_grad_(False)
                self.linear = self.linear.float()
        else:
            self.linear_mask = nn.Parameter(torch.concat( (torch.ones(d), torch.zeros(n-d)) ).to(device))
            self.linear_mask.requires_grad_(False)
        self.full_linear = full_linear
        self.skip_linear = skip_linear
        self.not_skip_linear_diagonal = not_skip_linear_diagonal
        self.multFactor = nn.Parameter(torch.tensor(0.5).to(device))
        self.multFactor.requires_grad = True
        self.device = device
    
    def reset_weights_rnd(self):
        self.multFactor = nn.Parameter(torch.rand(1))

    def set_factor(self, factor):
        self.multFactor = nn.Parameter(torch.tensor(float(factor)).to(self.device))

    def forward(self, x):
        if self.full_linear:
            return self.linear(x.float())
        if self.skip_linear:
            return (x.float() * self.linear_mask) * self.multFactor
        else:
            if self.not_skip_linear_diagonal:
                return x.float() * self.linear_diagonal
            else:
                x = self.linear(x.float())
                return x * self.multFactor