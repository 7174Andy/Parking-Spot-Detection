import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def preprocess(image, res=None):
    """
    Resizes, normalizes, and converts image to float32.
    """
    # resize image to model input size
    if res is not None:
        image = TF.resize(image, res)

    # convert image to float
    image = image.to(torch.float32) / 255

    # normalize image to default torchvision values
    image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    return image
