from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from np_fft_helper import calc_vorticity
from utils import AverageMeter

logger = getLogger()


def infer_vorticity(dataloader: DataLoader, model: nn.Module, device: str):
    HR_Z, LR_Z, SR_Z = [], [], []

    with torch.no_grad():
        for Xs, ys in dataloader:
            Preds = model(Xs.to(device)).cpu().numpy()
            Xs, ys = Xs.numpy(), ys.numpy()

            for UV, Z in zip([Xs, ys, Preds], [LR_Z, HR_Z, SR_Z]):
                assert len(UV.shape) == 4
                for uv in UV:
                    Z.append(calc_vorticity(uv))
    HR_Z = np.stack(HR_Z)
    LR_Z = np.stack(LR_Z)
    SR_Z = np.stack(SR_Z)

    return HR_Z, LR_Z, SR_Z


def calc_error_ratios_for_vorticity(
    dataloader: DataLoader, model: nn.Module, device: str, error_orders: list = [1]
):

    HR_Z, _, SR_Z = infer_vorticity(dataloader, model, device)

    assert HR_Z.shape == SR_Z.shape
    assert len(HR_Z.shape) == 3

    dict_loss = {}
    for error_order in error_orders:
        loss1 = np.mean(np.abs(HR_Z - SR_Z) ** error_order, axis=(1, 2))
        loss2 = np.mean(np.abs(HR_Z) ** error_order, axis=(1, 2))
        assert loss1.shape == loss2.shape

        loss = np.mean(loss1 / loss2)
        dict_loss[error_order] = loss

    return dict_loss


def calc_error_ratios_for_scalar(
    dataloader: DataLoader, model: nn.Module, device: str, error_order: int
) -> AverageMeter:

    _ = model.eval()
    total_loss = AverageMeter()

    with torch.no_grad():
        for Xs, ys in dataloader:
            Xs, ys = Xs.to(device), ys.to(device)
            preds = model(Xs)

            loss1 = torch.mean(torch.abs(ys - preds) ** error_order, dim=(1, 2, 3))
            loss2 = torch.mean(torch.abs(ys) ** error_order, dim=(1, 2, 3))

            loss = torch.mean(loss1 / loss2).item()  # average over batch dim
            total_loss.update(val=loss, n=Xs.shape[0])  # n == num of batches

    return total_loss


def calc_error_ratios_for_vector(
    dataloader: DataLoader, model: nn.Module, device: str, error_order: int
) -> AverageMeter:

    _ = model.eval()
    total_loss = AverageMeter()

    with torch.no_grad():
        for Xs, ys in dataloader:
            Xs, ys = Xs.to(device), ys.to(device)
            preds = model(Xs)

            if error_order == 1:
                norm1 = torch.sqrt(torch.sum((ys - preds) ** 2, dim=1))
                norm2 = torch.sqrt(torch.sum(ys ** 2, dim=1))

                loss1 = torch.mean(norm1, dim=(1, 2))  # average over space
                loss2 = torch.mean(norm2, dim=(1, 2))
            elif error_order == 2:
                loss1 = torch.mean((ys - preds) ** 2, dim=(1, 2, 3))
                loss2 = torch.mean((ys) ** 2, dim=(1, 2, 3))
            else:
                raise NotImplementedError()

            loss = torch.mean(loss1 / loss2).item()  # average over batch dim
            total_loss.update(val=loss, n=Xs.shape[0])  # n == num of batches

    return total_loss
