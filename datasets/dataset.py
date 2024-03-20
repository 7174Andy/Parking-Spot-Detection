import torch
from torch.utils.data import Dataset


class ParkingSpotDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
