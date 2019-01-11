import numbers
import numpy as np
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


def get_f1_np(preds, targs):
    tp = np.sum(preds * targs, axis=0).astype(np.float)
    tn = np.sum((1 - preds) * (1 - targs), axis=0).astype(np.float)
    fp = np.sum(preds * (1 - targs), axis=0).astype(np.float)
    fn = np.sum((1 - preds) * targs, axis=0).astype(np.float)

    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)

    f1 = 2*p*r / (p+r+1e-10)
    return np.mean(f1)


def stats_by_class(preds, targs):
    tp = np.sum(preds * targs, axis=0).astype(np.float)
    tn = np.sum((1 - preds) * (1 - targs), axis=0).astype(np.float)
    fp = np.sum(preds * (1 - targs), axis=0).astype(np.float)
    fn = np.sum((1 - preds) * targs, axis=0).astype(np.float)

    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)

    return p, r


def get_f1_threshold(preds, targs, th=0.0):
    if isinstance(th, list):
        # threshold by class
        th = np.array(th)

    if isinstance(th, np.ndarray) and len(th.shape) == 1:
        th = np.expand_dims(th, 0)

    preds = (preds > th)
    targs = (targs > th)

    return get_f1_np(preds, targs)


def get_f1_threshold_soft(preds, targs, th=0.0):
    if isinstance(th, list):
        # threshold by class
        th = np.array(th)

    if isinstance(th, np.ndarray) and len(th.shape) == 1:
        th = np.expand_dims(th, 0)

    def sigmoid_np(x):
        return 1.0 / (1.0 + np.exp(-x))

    b = 500.
    preds = sigmoid_np(b * (preds - th))
    targs = targs

    return get_f1_np(preds, targs)


def get_precision_soft(preds, targs, th=0.0):
    if isinstance(th, list):
        # threshold by class
        th = np.array(th)

    if isinstance(th, np.ndarray) and len(th.shape) == 1:
        th = np.expand_dims(th, 0)

    def sigmoid_np(x):
        return 1.0 / (1.0 + np.exp(-x))

    b = 100.
    preds = sigmoid_np(b * (preds - th))
    targs = targs

    return np.mean(stats_by_class(preds, targs)[0])
