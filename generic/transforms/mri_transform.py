# %%
import fastmri
import fastmri.data.transforms as mri_transform
import torch

from typing import (
    Optional, Union,
    Sequence, Tuple
)

from generic.transforms.mri_subsample import (
    MaskFunc,
    RandomMaskFunc
)

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """

    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed, device=data.get_device()) # changed here
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies

def generate_mask(data, padding=None, offset=None, seed=None, mask_func=None):
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed, device=data.get_device()) # changed here
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    #mask, num_low_frequencies = generate_mask_batched(data, padding=padding, offset=offset, seed=seed, mask_func=mask_func)

    mask = mask.to(data.get_device())

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies

def generate_mask(data, padding=None, offset=None, seed=None, mask_func=None):
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:]) 
    mask, num_low_frequencies = mask_func(shape, offset, seed, device=data.get_device()) # changed here
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
    return mask, num_low_frequencies


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask

# %%
rnd_mask_func = RandomMaskFunc(
    center_fractions=[0.08],
    accelerations=[4],
    )

# %%
def data_transform(kspace_torch, target, return_meta_data=False, fixed_mask=None, fixed_mean=None, fixed_std=None):
    crop_size = (320, 320)
    # if mask available
    if fixed_mask is not None:
        #print(f"Using fixed mask")
        masked_kspace = kspace_torch * fixed_mask + 0.0  # the + 0.0 removes the sign of the zeros
    elif rnd_mask_func:
        #print(f"Generating new mask")
        masked_kspace, mask, _ = apply_mask(kspace_torch, rnd_mask_func, seed=None) # last argument is number
    else:
        masked_kspace = kspace_torch

    sampled_image = fastmri.ifft2c(masked_kspace)
    sampled_image = mri_transform.complex_center_crop(sampled_image, crop_size)
    sampled_image = fastmri.complex_abs(sampled_image)
    
    # Apply Root-Sum-of-Squares if multicoil data
    #sampled_image = mri_transform.root_sum_of_squares(sampled_image)
    # Normalize input
    if fixed_mean is not None and fixed_std is not None:
        sampled_image = mri_transform.normalize(sampled_image, eps=1e-11, mean=fixed_mean, stddev=fixed_std)
        mean = fixed_mean
        std = fixed_std
    else:
        sampled_image, mean, std = mri_transform.normalize_instance(sampled_image, eps=1e-11)
    sampled_image = sampled_image.clamp(-6, 6)
    
    if target is not None:
        target_torch = mri_transform.center_crop(target, crop_size)
        target_torch = mri_transform.normalize(target_torch, mean, std, eps=1e-11)
        target_torch = target_torch.clamp(-6, 6)
    else:
        target_torch = torch.Tensor([0])

    if return_meta_data:
        return sampled_image, masked_kspace, target_torch, mask, mean, std
    else:
        return sampled_image, target_torch


def data_transform_measurement(kspace_torch, mask):
    crop_size = (320, 320)
    masked_kspace = kspace_torch * mask + 0.0  # the + 0.0 removes the sign of the zeros

    sampled_image = fastmri.ifft2c(masked_kspace)
    sampled_image = mri_transform.complex_center_crop(sampled_image, crop_size)
    sampled_image = fastmri.complex_abs(sampled_image)
    
    # Apply Root-Sum-of-Squares if multicoil data
    #sampled_image = mri_transform.root_sum_of_squares(sampled_image)
    # Normalize input
    sampled_image, mean, std = mri_transform.normalize_instance(sampled_image, eps=1e-11)
    sampled_image = sampled_image.clamp(-6, 6)
    
    return sampled_image, mean, std

def data_transform_target(target, mean, std):
    crop_size = (320, 320)
    target_torch = mri_transform.center_crop(target, crop_size)
    target_torch = mri_transform.normalize(target_torch, mean, std, eps=1e-11)
    target_torch = target_torch.clamp(-6, 6)
    return target_torch