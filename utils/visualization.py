import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def plot_image(image, boxes, labels):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box, label in zip(boxes, labels):
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        if label == 1:
            color = "r"
        else:
            color = "g"
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
