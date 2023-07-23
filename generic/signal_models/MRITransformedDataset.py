from torch.utils.data import Dataset
import fastmri

class MRITransformedDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, device="cpu"):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # raw: kspace : np.ndarray, mask : np.ndarray, target : np.ndarray , attrs : Dict, fname : str, slice_num : int
        kspace, mask, target, attr, filename, slice = self.dataset[idx]
        kspace = fastmri.data.transforms.to_tensor(kspace).unsqueeze(0).to(self.device)
        target = fastmri.data.transforms.to_tensor(target).unsqueeze(0).to(self.device)
        kspace_tranfs = self.transform(kspace) if self.transform != None else kspace
        target_transf = self.target_transform(target) if self.target_transform != None else target
        return kspace_tranfs, kspace, target_transf