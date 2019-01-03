import torch
from torch import nn
import torch.nn.functional as F
from theconf import Config as C
import pretrainedmodels

from common import num_class


class PNasnet(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet')
        conv0 = nn.Conv2d(4, 96, kernel_size=3, stride=2, bias=False)
        if pre:
            w = self.encoder.conv_0.conv.weight
            conv0.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.encoder.conv_0.conv = conv0
        self.dropout = nn.Dropout(C.get()['dropout'])
        self.last_linear = nn.Linear(4320, num_class())

    def forward(self, x):
        # x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear')
        x = self.encoder.features(x)
        # x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        x = torch.sigmoid(x)
        return x


class Nasnet(PNasnet):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')
        conv0 = nn.Conv2d(4, 96, kernel_size=3, stride=2, bias=False)
        if pre:
            w = self.encoder.conv0.conv.weight
            conv0.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.encoder.conv0.conv = conv0
        self.last_linear = nn.Linear(4032, num_class())


class Polynet(PNasnet):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained='imagenet')
        conv = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        if pre:
            w = self.encoder.stem.conv1[0].conv.weight
            conv.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.encoder.stem.conv1[0].conv = conv
        self.last_linear = nn.Linear(2048, num_class())


class SENet154(PNasnet):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
        conv = nn.Conv2d(4, 64, kernel_size=3, stride=2, bias=False)
        if pre:
            w = self.encoder.layer0[0].weight
            conv.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.encoder.layer0[0] = conv
        self.last_linear = nn.Linear(2048, num_class())
