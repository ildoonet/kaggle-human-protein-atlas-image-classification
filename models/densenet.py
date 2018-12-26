import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from theconf import Config as C
from common import num_class


class Densenet(nn.Module):
    def __init__(self):
        super().__init__()

    def _replace(self, pre, feat_size=1024):
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pre:
            w = self.encoder.features[0].weight
            conv1.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))
        self.encoder.features[0] = conv1
        self.encoder.classifier = nn.Linear(feat_size, num_class())
        self.dropout = nn.Dropout2d(C.get()['dropout'])

    def forward(self, x):
        x = self.encoder.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dropout(x)
        x = torch.squeeze(x)
        x = self.encoder.classifier(x)
        x = torch.sigmoid(x)

        return x


class Densenet121(Densenet):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = torchvision.models.densenet121(pretrained=pre)
        self._replace(pre)


class Densenet161(Densenet):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = torchvision.models.densenet161(pretrained=pre)
        self._replace(pre, 1664)


class Densenet169(Densenet):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = torchvision.models.densenet169(pretrained=pre)
        self._replace(pre, 1664)


class Densenet201(Densenet):
    def __init__(self, pre=True):
        super().__init__()
        self.encoder = torchvision.models.densenet201(pretrained=pre)
        self._replace(pre, 1664)
