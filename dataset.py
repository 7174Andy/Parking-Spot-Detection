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
        if ds_type in ["train", "val", "test"]:
            annotations = self.annotations[ds_type]
        else:
            assert ds_type == "all"
            annotations = {k: [] for k in all_annotations["train"].keys()}
            for ds_type in ["train", "valid", "test"]:
                for k, v in all_annotations[ds_type].items():
                    annotations[k] += v

        self.f_name = annotations["file_names"]
        self.rois_list = annotations["rois_list"]
        self.occupancy = annotations["occupancy_list"]

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
