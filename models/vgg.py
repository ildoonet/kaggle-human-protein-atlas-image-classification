import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from common import num_class


class Vgg16(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        encoder = torchvision.models.vgg16_bn(pretrained=pre)

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        if pre:
            w = encoder.features[0].weight
            self.conv1.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))

        self.layers = encoder.features[1:]
        self.fc = encoder.classifier[:-1]

        self.out = nn.Linear(4096, num_class())

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear')     # resize
        x = self.conv1(x)
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


# TODO : vgg13, vgg19
