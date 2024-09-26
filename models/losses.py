from kornia.filters import Sobel
import torch
from torch import nn


def l1_sobel_loss(y_true, y_pred):
    sobel = Sobel()
    l1 = nn.L1Loss(reduction='mean')

    sobel_loss = torch.mean(torch.abs(sobel(y_pred) - sobel(y_true)))
    l1_loss = l1(y_pred, y_true)

    return sobel_loss + l1_loss


def l2_sobel_loss(y_true, y_pred):
    sobel = Sobel()
    l2 = nn.MSELoss(reduction='mean')

    sobel_loss = torch.mean(torch.abs(sobel(y_pred) - sobel(y_true)))
    l2_loss = l2(y_pred, y_true)

    return sobel_loss + l2_loss
