import fiftyone as fo
import fiftyone.zoo as foz
import json
import os
import numpy as np
import shutil

output_path = "datasets/yolo-datasets/images"


# Load the images from the folder
def load_images_from_folder(folder, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_names = []
    for filename in os.listdir(folder):
        source = os.path.join(folder, filename)
        destination = f"{output_path}/{filename}"
        try:
            shutil.copy(source, destination)
            print(f"Copying {source} to {destination}")
        except shutil.SameFileError:
            print(f"Failed to copy {source} to {destination}")

        file_names.append(filename)

    return file_names


def get_img_annotation(img_id, data):
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
        splits=["validation", "train"],
        label_types=["detections"],
        dataset_dir="./datasets/coco-2017",
        max_samples=100,
    )

    # Visualize the dataset through the App
    session = fo.launch_app(dataset, port=5151)
    session.wait()

    session.dataset = dataset


def get_img_by_filename(data, filename):
    for img in data["images"]:
        if img["file_name"] == filename:
            return img


def convert_labels(json_dir, input_path, output_path):
    # Open the JSON file and load the data
    f = open(json_dir)
    data = json.load(f)
    f.close()

    file_names = load_images_from_folder(input_path + "/data", output_path + "/images")
    print("Finished loading images from the folder")

    for filename in file_names:
        # Extract images
        print(filename)
        image = get_img_by_filename(data=data, filename=filename)
        img_id = image["id"]
        img_width = image["width"]
        img_height = image["height"]

        # Get annotations
        img_annotation = get_img_annotation(
            img_id=img_id,
            data=data,
        )
        print(f"Processing {filename}...")

        if img_annotation:
            file_label = open(f"{output_path}/labels/{filename}.txt", "w")

            for ann in img_annotation:
                current_category = ann["category_id"] - 1
                current_bbox = ann["bbox"]
                x = current_bbox[0]
                y = current_bbox[1]
                box_width = current_bbox[2]
                box_height = current_bbox[3]

                # Get the center of the bounding box
                x_center = x + (x + box_width) / 2
                y_center = y + (y + box_height) / 2

                # Normalization
                x_center = x_center / img_width
                y_center = y_center / img_height
                box_width = box_width / img_width
                box_height = box_height / img_height

                # Limiting the values with 6 decimal figures
                x_center = format(x_center, ".6f")
                y_center = format(y_center, ".6f")
                box_width = format(box_width, ".6f")
                box_height = format(box_height, ".6f")

                # Write those data into the new label.txt file
                file_label.write(
                    f"{current_category} {x_center} {y_center} {box_width} {box_height}\n"
                )

            print(f"Finished writing {filename}.txt")
            file_label.close()


def main():
    # Download COCO Dataset
    download_coco_dataset()
    convert_labels(
        json_dir="datasets/coco-2017/validation/labels.json",
        input_path="datasets/coco-2017/validation",
        output_path="datasets/yolo-datasets/validation",
    )
    convert_labels(
        json_dir="datasets/coco-2017/train/labels.json",
        input_path="datasets/coco-2017/train",
        output_path="datasets/yolo-datasets/train",
    )


if __name__ == "__main__":
    main()
