import fiftyone as fo
import fiftyone.zoo as foz
import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

input_path = "datasets/coco-2017/validation"
output_path = "datasets/yolo-datasets/images"

file_names = []
f = open("datasets/coco-2017/validation/labels.json")
data = json.load(f)
f.close()


# Load the images from the folder
def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        source = os.path.join(folder, filename)
        destination = f"{output_path}/{filename}"
        try:
            shutil.copy(source, destination)
            print(f"Copying {source} to {destination}")
        except shutil.SameFileError:
            print(f"Failed to copy {source} to {destination}")

        file_names.append(filename)


def get_img_annotation(img_id):
    img_annotation = []
    isFound = False
    for ann in data["annotations"]:
        if ann["image_id"] == img_id:
            img_annotation.append(ann)
            isFound = True
    if isFound:
        return img_annotation
    else:
        return None


def download_coco_dataset():
    # Download the COCO 2017 Dataset
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["validation", "training"],
        label_types=["detections"],
        dataset_dir="./datasets/coco-2017",
        max_samples=1000,
    )

    # Visualize the dataset through the App
    session = fo.launch_app(dataset, port=5151)
    session.wait()

    session.dataset = dataset


def get_img_by_filename(filename):
    for img in data["images"]:
        if img["file_name"] == filename:
            return img


def main():
    print(data["images"][0])


if __name__ == "__main__":
    main()
