import torch
from torch import nn


class PSNR(nn.Module):
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val_squared = max_val ** 2

    def forward(self, outputs, targets):
        """
        outputs: model predictions
        targets: ground truths
        """

        mse = torch.mean((outputs - targets) ** 2, dim=(1, 2, 3))  # batch dim (= 0) remains
        psnr = 10.0 * torch.log10(self.max_val_squared / mse)
        psnr = torch.mean(psnr)  # average over batch dim

        return psnr
