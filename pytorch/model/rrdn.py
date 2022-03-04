import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, negative_slope: float
    ):
        super(DenseLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        return torch.cat(
            [x, self.lrelu(self.conv(x))], 1
        )  # `1` means concatenating along the channel dimension


class DenseBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        kernel_size: int,
        beta: float,
        negative_slope: float,
    ):
        super(DenseBlock, self).__init__()

        self.beta = beta

        self.layers = nn.Sequential(
            *[
                DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size, negative_slope)
                for i in range(num_layers)
            ]
        )

        # local feature fusion (lff)
        self.lff = nn.Conv2d(
            in_channels + growth_rate * num_layers,
            in_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        return x + self.beta * self.lff(self.layers(x))  # local residual learning
        # The size of output is the same as that of input


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB)"""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        kernel_size: int,
        num_blocks: int,
        beta: float,
        negative_slope: float,
    ):
        super(RRDB, self).__init__()

        self.blocks = nn.Sequential(
            *[
                DenseBlock(in_channels, growth_rate, num_layers, kernel_size, beta, negative_slope)
                for _ in range(num_blocks)
            ]
        )

        self.beta = beta

    def forward(self, x):
        return x + self.beta * self.blocks(x)


class RRDN(nn.Module):
    """Residual in Residual Dense Network (RRDN) proposed by Bode et al. (2021) as the generator of their PIESRGAN.
    Ref1) https://doi.org/10.1016/j.proci.2020.06.022
    Ref2) https://git.rwth-aachen.de/Mathis.Bode/PIESRGAN/-/blob/master/PIESRGAN.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        num_features: int = 64,
        post_features: int = 512,
        post_slope: float = 0.01,
        growth_rate: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        num_blocks: int = 3,
        beta: float = 0.2,
        negative_slope: float = 0.01,
    ):
        super(RRDN, self).__init__()

        self.beta = beta
        padding = kernel_size // 2

        # shallow feature extractor (sfe)
        self.sfe_conv = nn.Conv2d(in_channels, num_features, kernel_size, padding=padding)
        self.sfe_lrelu = nn.LeakyReLU(negative_slope, inplace=True)

        self.rrdb = RRDB(
            num_features, growth_rate, num_layers, kernel_size, num_blocks, beta, negative_slope
        )

        # global feature fusion (gff)
        self.gff = nn.Conv2d(num_features, num_features, kernel_size, padding=padding)

        # post process (pp)
        self.pp_layers = nn.Sequential(
            nn.Conv2d(num_features, post_features, kernel_size, padding=padding),
            nn.LeakyReLU(post_slope, inplace=True),
            nn.Conv2d(post_features, post_features, kernel_size, padding=padding),
            nn.LeakyReLU(post_slope, inplace=True),
            nn.Conv2d(post_features, num_features, kernel_size, padding=padding),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size, padding=padding),
        )

    def forward(self, x):
        f = self.sfe_lrelu(self.sfe_conv(x))
        y = self.rrdb(f)
        y = f + self.beta * self.gff(y)
        return self.pp_layers(y)
