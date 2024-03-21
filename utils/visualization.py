import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def draw_boxes(image, boxes, class_labels, class_names, colors):
    """
    Draw bounding boxes around an image.

    :param image: NumPy array image in RGB format.
    :param boxes: NMS appied bounding boxes. Shape is [N, 6].
        Normalized box coordinates start from index 2 in the format of
        [x_center, y_center, normalized width, normalized height].

    Returns:
        image: NumPy array image with bounding boxes drawn.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Get the original image height and width.
    for i, box in enumerate(boxes):
        color = colors[int(class_labels[i])][::-1]
        class_name = class_names[int(class_labels[i])]
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            image,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(class_name),
            (int(x_min), int(y_min) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return image


def yolo2bbox(bboxes, width, height):
    """
    Function to convert bounding boxes in YOLO format to
    xmin, ymin, xmax, ymax.

    Parmaeters:
    :param bboxes: Normalized [x_center, y_center, width, height] list
    :param width: Original width of the image.
    :param height: Original height of the image.

    return: xmin, ymin, xmax, ymax relative to original image size.
    """
    xmin, ymin = (bboxes[0] - bboxes[2] / 2) * width, (
        bboxes[1] - bboxes[3] / 2
    ) * height
    xmax, ymax = (bboxes[0] + bboxes[2] / 2) * width, (
        bboxes[1] + bboxes[3] / 2
    ) * height
    return xmin, ymin, xmax, ymax
