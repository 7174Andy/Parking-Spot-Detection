import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2


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

    def to_prediction_tensor(self, bboxes, labels):
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box, label in zip(bboxes, labels):
            class_label = label
            x, y, w, h = box.tolist()
            class_label = int(class_label)

            # Convert to grid cell coordinates.
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # Convert to width and height coordinates.
            width_cell, height_cell = (
                w * self.S,
                h * self.S,
            )

            # If the cell is already occupied, we don't want to overwrite it.
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell], dtype=torch.float32
                )
                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return label_matrix

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = cv2.imread(os.path.join(self.data_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = self.bboxes[idx]
        labels = self.labels[idx]

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, bboxes, labels


def __main__():
    data_dir = "data/Parking Space.v4i.darknet/train"
    dataset = ParkingSpotDataset(image_size=448, data_dir=data_dir)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1
    )


if __name__ == "__main__":
    __main__()
