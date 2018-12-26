import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models.inception import BasicConv2d, InceptionAux

from common import num_class


class InceptionV3(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = torchvision.models.inception_v3(pretrained=pre)

        conv1 = BasicConv2d(4, 32, kernel_size=3, stride=2)
        if pre:
            w = self.encoder.Conv2d_1a_3x3.conv.weight
            conv1.conv.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.encoder.Conv2d_1a_3x3 = conv1
        self.encoder.AuxLogits = InceptionAux(768, num_class())
        self.encoder.fc = nn.Linear(2048, num_class())

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear')     # resize
        if self.training:
            x, x_aux = self.encoder(x)
            x = (torch.sigmoid(x) + torch.sigmoid(x_aux)) * 0.5
        else:
            x = self.encoder(x)
            x = torch.sigmoid(x)

        return x
