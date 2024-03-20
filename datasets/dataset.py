import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2


class ParkingSpotDataset(Dataset):
    def __init__(self, data_dir, S=7, B=2, C=2, transform=None):
        self.data_dir = data_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

        self.bboxes = []
        self.labels = []
        self.image_names = []

        files = os.listdir(data_dir)
        label_files = [file for file in files if os.path.splitext(file)[1] == ".txt"]

        for label_file in label_files:
            bbox = []
            labels = []
            image_name = os.path.splitext(label_file)[0] + ".jpg"
            with open(os.path.join(data_dir, label_file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_label, x, y, w, h = map(float, line.strip().split())
                    bbox.append([x, y, w, h])
                    labels.append(int(class_label))

            self.bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def to_prediction_tensor(self, bboxes, labels):
        prediction_matrix = torch.zeros(
            (self.S, self.S, self.C + 5 * self.B), dtype=torch.float32
        )
        i = 0
        box = bboxes[i]
        class_label = labels[i]
        width, height = box[2:] - box[:2]
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        i, j = int(self.S * y), int(self.S * x)
        width_cell, height_cell = self.S * width, self.S * height

        if prediction_matrix[i, j, 0] == 0:
            prediction_matrix[i, j, 0] = 1
            prediction_matrix[i, j, 1:5] = torch.tensor([x, y, width, height])
            prediction_matrix[i, j, 5 + class_label] = 1

        return prediction_matrix

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(os.path.join(self.data_dir, image_name))
        bboxes = self.bboxes[idx]
        labels = self.labels[idx]

        target = self.to_prediction_tensor(bboxes, labels)

        return image, target


def __main__():
    dataset = ParkingSpotDataset(data_dir="data/Parking Space.v4i.darknet/train")
    print(len(dataset))


if __name__ == "__main__":
    __main__()
