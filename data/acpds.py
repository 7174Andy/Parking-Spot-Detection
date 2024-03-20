import torch
import json
import os
import numpy as np
import shutil
import torchvision
import torchvision.transforms.functional as TF


from torch.utils.data import DataLoader
from torchvision import transforms


class ACPDS:
    """
    ACPDS dataset class
    """

    def __init__(self, dataset_path, ds_type="train", res=None):
        self.dataset_path = dataset_path
        self.ds_type = ds_type
        self.res = res

        with open(os.path.join(dataset_path, "annotations.json"), "r") as f:
            all_annotations = json.load(f)

        # select the dataset type
        if ds_type in ["train", "valid", "test"]:
            annotations = all_annotations[ds_type]
        else:
            assert ds_type == "all"
            annotations = {k: [] for k in all_annotations["train"].keys()}
            for ds_type in ["train", "valid", "test"]:
                for k, v in all_annotations[ds_type].items():
                    annotations[k] += v

        self.f_name = annotations["file_names"]
        self.rois_list = annotations["rois_list"]
        self.occupancy = annotations["occupancy_list"]

    def convert_to_yolo(self, rois, img_size):
        """
        Convert rois to yolo format
        """
        # rois: [x, y, w, h]
        # img_size: [h, w]
        x, y, w, h = rois
        x_c = x + w / 2
        y_c = y + h / 2
        return x_c / img_size[1], y_c / img_size[0], w / img_size[1], h / img_size[0]

    def __len__(self):
        return len(self.f_name)

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.dataset_path, "images", self.f_name[idx])
        img = torchvision.io.read_image(img_path)
        if self.res is not None:
            img = TF.resize(img, self.res)

        # load occupancy
        occupancy = self.occupancy[idx]
        occupancy = torch.tensor(occupancy, dtype=torch.float32)

        # load rois
        rois = self.rois_list[idx]
        rois = torch.tensor(rois, dtype=torch.float32)

        return img, rois, occupancy


def collate_fn(batch):
    images = [item[0] for item in batch]
    rois = [item[1] for item in batch]
    occupancy = [item[2] for item in batch]
    return [images, rois, occupancy]


def create_datasets(dataset_path, *args, **kwargs):
    """
    Create training and test DataLoaders.
    Returns the tuple (image, rois, occupancy).
    During the first pass, the DataLoaders will be cached.
    """
    ds_train = ACPDS(dataset_path, "train", *args, **kwargs)
    ds_valid = ACPDS(dataset_path, "valid", *args, **kwargs)
    ds_test = ACPDS(dataset_path, "test", *args, **kwargs)
    data_loader_train = DataLoader(
        ds_train, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    data_loader_valid = DataLoader(
        ds_valid, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    data_loader_test = DataLoader(
        ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    return data_loader_train, data_loader_valid, data_loader_test
