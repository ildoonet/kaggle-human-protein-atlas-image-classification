import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models.inception import BasicConv2d, InceptionAux
import pretrainedmodels

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
            x, x_aux, feat = self.encoder(x)
            x = (torch.sigmoid(x) + torch.sigmoid(x_aux)) * 0.5
        else:
            x, feat = self.encoder(x)
            x = torch.sigmoid(x)

        return {'logit': x, 'feat': feat}


class InceptionV4(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        conv1 = BasicConv2d(4, 32, kernel_size=3, stride=2)
        if pre:
            w = self.encoder.features[0].conv.weight
            conv1.conv.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.encoder.features[0].conv = conv1
        self.last_linear = nn.Linear(1536, num_class())
        pass

    def forward(self, x):
        # x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear')
        x = self.encoder.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        feat = x
        x = self.last_linear(x)
        x = torch.sigmoid(x)
        return {'logit': x, 'feat': feat}
