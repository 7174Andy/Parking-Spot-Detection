import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as T
import numpy as np

class DarkNet(nn.Module):
    def __init__(self, pretrained=False):
        super(DarkNet, self).__init__()
        
        