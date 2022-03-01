from logging import getLogger
from typing import Tuple

import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from loss import PSNR
from utils import AverageMeter

logger = getLogger()


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    tqdm_message: str = None,
    max_val: float = 2.0,
) -> Tuple[float, float, float]:

    data_size = len(dataloader.dataset)

    mse_fn, psnr_fn = nn.MSELoss(), PSNR(max_val)
    train_loss, train_mse, train_psnr = AverageMeter(), AverageMeter(), AverageMeter()

    model.train()

    with tqdm(total=data_size) as t:
        if tqdm_message is not None:
            t.set_description(tqdm_message)

        for Xs, ys in dataloader:
            Xs, ys = Xs.to(device), ys.to(device)
            preds = model(Xs)
            loss = loss_fn(preds, ys)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(Xs))
            train_mse.update(mse_fn(preds, ys).item(), n=len(Xs))
            train_psnr.update(psnr_fn(preds, ys).item(), n=len(Xs))

            t.update(len(Xs))

    logger.info(
        f"Train error: avg loss = {train_loss.avg:.8f}, avg mse = {train_mse.avg:.8f}, avg psnr = {train_psnr.avg:.4f}"
    )

    return train_loss.avg, train_mse.avg, train_psnr.avg


def train_ddp_model(
    model: nn.Module,
    loss_fn: nn.functional,
    dataloader: DataLoader,
    sampler: Sampler,
    epoch: int,
    rank: int,
    world_size: int,
    optimizer: Optimizer,
) -> float:

    model.train()
    sampler.set_epoch(epoch)
    data_size = len(dataloader.dataset)
    sum_loss = 0.0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(rank), y.to(rank)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss * len(X)

        if (batch_idx + 1) % 10 == 0 and rank == 0:
            progress_count = (batch_idx + 1) * len(X) * world_size
            progress_msg = f"Train: epoch = {epoch}, {progress_count}/{data_size}"
            logger.info(progress_msg)

    dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
    mean_loss = sum_loss.item() / data_size

    if rank == 0:
        result_msg = f"Train: epoch = {epoch}, mean loss = {mean_loss:.8f}"
        logger.info(result_msg)

    return mean_loss
