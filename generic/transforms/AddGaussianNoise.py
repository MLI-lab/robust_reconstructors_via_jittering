import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        device_ind = tensor.get_device()
        if device_ind == -1:
            device_ind = "cpu"
        return tensor + torch.randn(tensor.size(), device=device_ind) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)