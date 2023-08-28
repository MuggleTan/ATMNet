import numpy as np
import torch
import torch.nn.functional as f
from torch import nn


def mse_loss_func(pred, gt, mask):
    return f.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    return f.l1_loss(pred[mask == 1.], gt[mask == 1.])


def weighted_l1_loss_func(pred, gt, mask, loss_weight):
    return torch.sum(torch.abs(pred - mask)[mask == 1.] * loss_weight[mask == 1.])
