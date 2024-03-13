import fiftyone as fo
import fiftyone.zoo as foz
import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

input_path = "datasets/coco-2017"
output_path = "datasets/yolo-datasets"

file_names = []


def load_images_from_folder(folder):
    count = 0
    for filename in os.listdir(folder):
        source = os.path.join(folder, filename)
        destination = f"{output_path}/images/{filename}"

        try:
            shutil.copy(source, destination)
            print(f"Image {filename} copied")
        except:
            print(f"Image {filename} already exists")
            continue

        file_names.append(filename)
        count += 1


def download_coco_dataset():
    # Download the COCO 2017 Dataset
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        dataset_dir="./datasets/coco-2017",
        max_samples=1000,
    )

    # Visualize the dataset through the App
    session = fo.launch_app(dataset, port=5151)
    session.wait()

    session.dataset = dataset


def main():
    load_images_from_folder("datasets/coco-2017/validation")
    print(file_names)


if __name__ == "__main__":
    main()
