import torch
from torch import nn
from torch.nn import functional as F


def acc(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


def f1_loss(preds, targs):
    return 1 - get_f1(preds, targs)


def get_f1(preds, targs):
    tp = torch.sum(preds * targs, dim=0).float()
    tn = torch.sum((1 - preds) * (1 - targs), dim=0).float()
    fp = torch.sum(preds * (1 - targs), dim=0).float()
    fn = torch.sum((1 - preds) * targs, dim=0).float()

    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)

    f1 = 2*p*r / (p+r+1e-10)
    return torch.mean(f1)


def stats_by_class(preds, targs):
    tp = preds * targs
    tn = torch.sum((1 - preds) * (1 - targs), dim=0).float()
    fp = torch.sum(preds * (1 - targs), dim=0).float()
    fn = torch.sum((1 - preds) * targs, dim=0).float()
    return []   # TODO


def get_f1_threshold(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = (targs > th).int()
    return get_f1(preds, targs)


# TODO : 'categorical_accuracy', 'binary_accuracy'
