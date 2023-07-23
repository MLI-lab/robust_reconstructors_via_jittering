import os
import tqdm
from glob import glob
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision
from PIL import Image
import math
import random

from generic.signal_models.TensorListDataset import TensorListDataset

def read_image(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def transform_normlizations_via_config(transform_normalize):
    means = transform_normalize["mean"]
    stds = transform_normalize["std"]
    normalization = transforms.Normalize(mean=means,std=stds)
    means_inv = list(map(lambda x: -x[0]/x[1], zip(means, stds)))
    stds_inv = list(map(lambda s: 1.0/s, stds))
    normalization_inv = transforms.Normalize(mean=means_inv,std=stds_inv)
    return normalization, normalization_inv

def transform_stack_via_config(transform_random_crop, transform_grayscale, transform_normalize):
    print(transform_normalize)
    transf_list = []
    if transform_random_crop["enabled"]:
        transf_list.append(transforms.RandomCrop( (transform_random_crop["width"], transform_random_crop["height"]) ))

    transform_stack = transf_list + [transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()]

    if transform_grayscale:
        transform_stack.append(transforms.Grayscale())
    
    if transform_normalize["enabled"]:
        normalization, _ = transform_normlizations_via_config(transform_normalize)
        transform_stack.append(normalization)

    return transforms.Compose(transform_stack)

def dataset_via_config(config, device=None):
    if config["output_format"]["tensors"]:
        return TensorListDataset(config["path_to"], device=device, single_tensor=config["output_format"]["single_tensor"])
    else:
        return torchvision.datasets.ImageFolder(config["path_to"], transform_stack_via_config(config["transform_random_crop"], config["transform_grayscale"], config["transform_normalize"]))

def generate_dataset_single(
    path_from,
    path_to,
    file_glob_append,
    subset_settings =  {"fraction_start" : 0.0, "fraction_stop" : 1.0, "number_samples" : 5000, "random_subset" : True, "random_seed" : 0},
    transform_normalize = {"enabled" : True, "mean" : [0.454], "std" : [0.199]},
    transform_random_crop =  {"enabled" : False},
    transform_grayscale =  False,
    output_format = {"tensors" : True, "single_tensor" : True},
    filter  = {"enabled" : False, "reps" : 1, "center": 1000, "width": 500}, devices = ["cpu"]):

    print(f"Generating from {path_from} to {path_to}")

    if not os.path.exists(path_from):
        print(f"Data directory does not exist: {path_from}!")
        return
 
    # need inverse normalization if output format are images again
    if not output_format["tensors"]:
        _, normalization_inv = transform_normlizations_via_config(transform_normalize)

    # build transform stack
    transform_stack = transform_stack_via_config(transform_random_crop, transform_grayscale, transform_normalize)

    files_full = glob(os.path.join(path_from, file_glob_append))
    file_nr = len(files_full)

    print(f"Has: {file_nr} files in total.")
    start_files = int(math.floor(subset_settings["fraction_start"]*file_nr))
    stop_files = int(math.floor(subset_settings["fraction_stop"]*file_nr))
    if stop_files < start_files + subset_settings["number_samples"]:
        print(f"Fractions yield {stop_files-start_files} but {subset_settings['number_samples']} specified.")

    files = files_full[start_files:stop_files]
    if subset_settings['number_samples'] < stop_files - start_files:
        print(f"Random sampling {subset_settings['number_samples']} of {stop_files-start_files} elements.")
        random.seed(subset_settings['random_seed'])
        files = random.sample(files, subset_settings['number_samples'])

    if not os.path.exists(path_to):
        os.makedirs(path_to)

    img_nr_of_candidates = 0
    imgs_skipped = 0
    nr_reps = filter["reps"] if "reps" in filter else 1
    tensors = []
    for file in tqdm.tqdm(files):
        image = read_image(file)

        if transform_random_crop["enabled"]:
            if image.size[0] < transform_random_crop["height"] or image.size[1] < transform_random_crop["width"]:
                imgs_skipped += 1
                continue

        for rep in range(nr_reps):
            img_out = transform_stack(image)

            img_out_np = img_out.numpy()

            norm_square = np.inner(img_out_np.flatten(), img_out_np.flatten())
            if( (not filter["enabled"]) or (norm_square > filter["center"] - filter["width"]) and (norm_square < filter["center"] + filter["width"])):
                if output_format["tensors"]:
                    if output_format["single_tensor"]:
                        tensors.append(torch.unsqueeze(img_out, dim=0))
                    else:
                        new_file_path = os.path.join(path_to, f"tensor_{img_nr_of_candidates}.pt")
                        torch.save(img_out, new_file_path)
                    img_nr_of_candidates += 1
                else:
                    # retain dataset structure
                    new_file_path = os.path.join(path_to, os.path.relpath(file, path_from))
                    new_file_path_dir = os.path.dirname(new_file_path)
                    if not os.path.exists(new_file_path_dir):
                        os.makedirs(new_file_path_dir)
                    elif os.path.exists(new_file_path):
                        os.remove(new_file_path)

                    inv_img = normalization_inv(img_out)
                    jpg_img = (inv_img*255).to(torch.uint8)
                    torchvision.io.write_jpeg(jpg_img, new_file_path)
                    img_nr_of_candidates += 1
                    break # at max only one transformed image per base image

    print(f"Total {len(files)}; saved: {img_nr_of_candidates}; skipped: {imgs_skipped}")    
    if output_format["tensors"] and output_format["single_tensor"]:
        path_tensors = os.path.join(path_to, "tensors.pt")
        print(f"Saved list of tensors at: {path_tensors}")
        tensors_packed = torch.cat(tensors)
        torch.save(tensors_packed, path_tensors)