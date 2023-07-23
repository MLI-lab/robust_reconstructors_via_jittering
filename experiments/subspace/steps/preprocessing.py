from generic.signal_models.NoiseReconstructionDataset import NoiseReconstructionDataset
from generic.signal_models.UcSubspaceDataset import UcSubspaceDataset
from generic.transforms.AddGaussianNoise import AddGaussianNoise
from generic.transforms.LinearTransformation import create_diagonal_by_name, LinearTransformation
import torchvision.transforms as transforms

def dataset_via_config(config, device=None, noise_level=None):
    dataset_base = UcSubspaceDataset(n=config["n"], d=config["d"], size=config['size'], device=device)
    noise_level_used = noise_level if noise_level is not None else config["noise_level"]
    if config["linear_forward_name"] != "None":
        matrix_data = create_diagonal_by_name(config["linear_forward_name"], config["n"], config["linear_forward_kwargs"])
        return NoiseReconstructionDataset(dataset=dataset_base,
            transform=transforms.Compose([
                LinearTransformation(matrix_data, True, device),
                AddGaussianNoise(mean=0, std=noise_level_used)
                ]))

    else:
        return NoiseReconstructionDataset(dataset=dataset_base,
            transform=AddGaussianNoise(mean=0, std=noise_level_used))
