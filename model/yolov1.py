import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as T
import numpy as np
from torchvision import models
from torch.utils import model_zoo

# from detectron2 import model_zoo


# Utilizing the DarkNet architecture as a base for YOLOv1
class YOLOv1(nn.Module):
    def __init__(self, S, B, C):
        super(YOLOv1, self).__init__()
        # Initializing the pre-trained weights, using VGG11, for YOLOv1 arthitecture
        backbone = models.vgg11_bn(pretrained=True)
        # weights = model_zoo.load_url(
        #     url="https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        #     progress=True,
        # )
        # self.backbone.load_state_dict(state_dict=weights, strict=False)
        self.backbone = nn.Sequential(*list(backbone.features.children())[:-1])

        # Learning parameters for parking space
        self.yolo_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, S * S * (C + B * 5)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.yolo_head(x)
        return x

    # Initializaing random weights
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0]
                )
                m.weight.data.copy_(initial_weight)
