import torch
import torch.nn as nn
import math
import numpy as np

def create_kernel_by_name(name, channels):
    if name == "gaussian_4x4_s2_v1":
        kernel = np.array([[0.0479, 0.0615, 0.0615, 0.0479],
          [0.0615, 0.0790, 0.0790, 0.0615],
          [0.0615, 0.0790, 0.0790, 0.0615],
          [0.0479, 0.0615, 0.0615, 0.0479]], np.float32)
    elif name == "gaussian_8x8_s2_v1":
        kernel = np.array([[0.0020, 0.0043, 0.0071, 0.0091, 0.0091, 0.0071, 0.0043, 0.0020],
          [0.0043, 0.0091, 0.0150, 0.0193, 0.0193, 0.0150, 0.0091, 0.0043],
          [0.0071, 0.0150, 0.0248, 0.0318, 0.0318, 0.0248, 0.0150, 0.0071],
          [0.0091, 0.0193, 0.0318, 0.0408, 0.0408, 0.0318, 0.0193, 0.0091],
          [0.0091, 0.0193, 0.0318, 0.0408, 0.0408, 0.0318, 0.0193, 0.0091],
          [0.0071, 0.0150, 0.0248, 0.0318, 0.0318, 0.0248, 0.0150, 0.0071],
          [0.0043, 0.0091, 0.0150, 0.0193, 0.0193, 0.0150, 0.0091, 0.0043],
          [0.0020, 0.0043, 0.0071, 0.0091, 0.0091, 0.0071, 0.0043, 0.0020]], np.float32)
    elif name == "sharp":
        sharp_kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]], np.float32)
        kernel = sharp_kernel / np.sum(sharp_kernel)
    elif name == "outline":
        outline_kernel = np.array([[-1, -1, -1],[-1, 5, -1], [-1, -1, -1]], np.float32)
        kernel =  outline_kernel / np.sum(outline_kernel)
    elif name == "asym":
        asym_kernel = np.array([[0, 1, 0.25],[0, 1, 0.5], [-1,0,1]], np.float32)
        kernel =  asym_kernel / np.sum(asym_kernel)
    else:
        kernel = np.array([[1]], np.float32)
    grayscale_kernel =  create_fixed_grayscale_kernel(kernel)
    return grayscale_kernel.repeat(channels, *([1]*(len(grayscale_kernel.shape)-1)))

def create_fixed_grayscale_kernel(array):
    kernel = torch.from_numpy(array)
    return kernel.view(1, 1, array.shape[0], array.shape[1]).repeat(1,1,1,1)

class ApplyConvolution(object):
    def __init__(self, kernel, channels=1, device="cpu"):
        self._kernel = kernel.to(device)
        self._conv_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
            kernel_size=kernel.shape, stride=1, bias=False, padding="same", groups=channels)
        self._conv_filter.weight.data = self._kernel
        self._conv_filter.weight.requires_grad = False
        
    def __call__(self, tensor):
        if len(tensor.shape) < len(self._conv_filter.weight.data.shape):
            t = torch.unsqueeze(tensor, dim=0)
            ret = torch.squeeze(self._conv_filter(t), dim=0)
            return ret
        else:
            return self._conv_filter(tensor)
    
    def __repr__(self):
        return self.__class__.__name__