from logging import getLogger
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Sampler

from loss import PSNR
from utils import AverageMeter

logger = getLogger()


def validate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    device: str,
    max_val: float = 2.0,
) -> Tuple[float, float, float]:

    mse_fn, psnr_fn = nn.MSELoss(), PSNR(max_val)
    val_loss, val_mse, val_psnr = AverageMeter(), AverageMeter(), AverageMeter()

    model.eval()

    with torch.no_grad():
        for Xs, ys in dataloader:
            Xs, ys = Xs.to(device), ys.to(device)
            preds = model(Xs)

            val_loss.update(loss_fn(preds, ys).item(), n=len(Xs))
            val_mse.update(mse_fn(preds, ys).item(), n=len(Xs))
            val_psnr.update(psnr_fn(preds, ys).item(), n=len(Xs))

    logger.info(
        f"Validation error: avg loss = {val_loss.avg:.8f}, avg mse = {val_mse.avg:.8f}, avg psnr = {val_psnr.avg:.4f}"
    )

    return val_loss.avg, val_mse.avg, val_psnr.avg


def validate_ddp_model(
    model: nn.Module,
    loss_fn: nn.functional,
    dataloader: DataLoader,
    sampler: Sampler,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    model.eval()
    sampler.set_epoch(epoch)
    data_size = len(dataloader.dataset)
    sum_loss = 0.0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(rank), y.to(rank)
            pred = model(X)
            loss = loss_fn(pred, y)
            sum_loss += loss * len(X)

    dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
    mean_loss = sum_loss.item() / data_size

    if rank == 0:
        result_msg = f"Validation: epoch = {epoch}, mean loss = {mean_loss:.8f}"
        logger.info(result_msg)

    return mean_loss
