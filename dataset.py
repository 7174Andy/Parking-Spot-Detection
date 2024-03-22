import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class ParkingSpotDataset(Dataset):
    def __init__(self, image_size, data_dir, S=7, B=2, C=2, transform=None):
        self.data_dir = data_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.image_size = image_size

        self.bboxes = []
        self.labels = []
        self.image_names = []

        files = os.listdir(data_dir)
        label_files = [file for file in files if os.path.splitext(file)[1] == ".txt"]

        for label_file in label_files:
            bbox = []
            labels = []
            image_name = os.path.splitext(label_file)[0] + ".jpg"
            self.image_names.append(image_name)
            with open(os.path.join(data_dir, label_file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_label, x, y, w, h = map(float, line.strip().split())
                    bbox.append([x, y, w, h])
                    labels.append(int(class_label))

            self.bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def to_label_matrix(self, image, bboxes, labels):
        image = cv2.resize(image, (self.image_size, self.image_size))

        # plt.imshow(image)
        # print(image.shape)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))

        for box, label in zip(bboxes, labels):
            # Extracting the x, y, h, w values from the box
            x1, y1, h, w = box
            cls = label

            loc = [self.S * x1, self.S * y1]
            loc_i, loc_j = int(loc[1]), int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j

            if label_matrix[loc_i, loc_j, self.C] == 0:
                label_matrix[loc_i, loc_j, self.C] = 1
                label_matrix[loc_i, loc_j, self.C + 1 : self.C + 5] = torch.tensor(
                    [x, y, h, w]
                )
                label_matrix[loc_i, loc_j, cls] = 1

        return label_matrix

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = cv2.imread(os.path.join(self.data_dir, image_name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        bboxes = self.bboxes[idx].clone()
        labels = self.labels[idx].clone()

        label_matrix = self.to_label_matrix(image, bboxes, labels)

        return image, label_matrix


def __main__():
    data_dir = "data/Parking Space.v4i.darknet/train"
    dataset = ParkingSpotDataset(image_size=448, data_dir=data_dir)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1
    )


if __name__ == "__main__":
    __main__()
