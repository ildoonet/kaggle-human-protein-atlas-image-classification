import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from theconf import Config as C

from common import num_class


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(self.dropout(x))

        x = self.out(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.sigmoid(x)
        x = torch.squeeze(x)

        return x

    def set(self, encoder, feat_size, pre):
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pre:
            w = encoder.conv1.weight
            self.conv1.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.dropout = nn.Dropout2d(C.get()['dropout'])

        self.out = nn.Conv2d(feat_size, num_class(), kernel_size=3, padding=1)


class Resnet34(Resnet):
    def __init__(self, pre=True):
        super().__init__()
        encoder = torchvision.models.resnet34(pretrained=pre)
        self.set(encoder, 512, pre)


class Resnet50(Resnet):
    def __init__(self, pre=True):
        super().__init__()
        encoder = torchvision.models.resnet50(pretrained=pre)
        self.set(encoder, 2048, pre)


class Resnet101(Resnet):
    def __init__(self, pre=True):
        super().__init__()
        encoder = torchvision.models.resnet101(pretrained=pre)
        self.set(encoder, 2048, pre)


class Resnet152(Resnet):
    def __init__(self, pre=True):
        super().__init__()
        encoder = torchvision.models.resnet152(pretrained=pre)
        self.set(encoder, 2048, pre)
