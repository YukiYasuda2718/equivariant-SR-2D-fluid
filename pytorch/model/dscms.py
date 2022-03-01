from logging import getLogger

import torch
from torch import nn

logger = getLogger()


class DSCMS(nn.Module):
    """Hybrid Downsampled Skip-Connection/Multi-Scale model proposed by Fukami et al. (2019, JFM).
    Ref: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
    """

    def __init__(self, in_channels: int, out_channels: int, factor_filter_num: int):
        super(DSCMS, self).__init__()

        # Down-sampled skip-connection model (DSC)
        f_num1 = int(factor_filter_num * 32)
        logger.info(f"f_num1 = {f_num1} / 32, factor = {factor_filter_num}")

        self.dsc1_mp = nn.MaxPool2d(kernel_size=8, padding=0)
        self.dsc1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        # Regarding `algin_corners=False`, see the below
        # https://qiita.com/matsxxx/items/fe24b9c2ac6d9716fdee
        # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/20

        self.dsc2_mp = nn.MaxPool2d(kernel_size=4, padding=0)
        self.dsc2_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.dsc3_mp = nn.MaxPool2d(kernel_size=2, padding=0)
        self.dsc3_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.dsc4_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Multi-scale model (MS)
        f_num2 = int(factor_filter_num * 8)
        logger.info(f"f_num2 = {f_num2} / 8, factor = {factor_filter_num}")

        self.ms1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.ms2_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )

        self.ms3_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
        )

        self.ms4_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=(f_num2 * 3 + in_channels),
                out_channels=f_num2,
                kernel_size=7,
                padding=3,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # After concatenating DSC and MS
        self.final_layers = nn.Conv2d(
            in_channels=f_num1 + f_num2, out_channels=out_channels, kernel_size=3, padding=1
        )

    def _dsc(self, x):
        x1 = self.dsc1_layers(self.dsc1_mp(x))
        mp2 = self.dsc2_mp(x)
        x2 = self.dsc2_layers(torch.cat([x1, mp2], dim=1))
        mp3 = self.dsc3_mp(x)
        x3 = self.dsc3_layers(torch.cat([x2, mp3], dim=1))
        return self.dsc4_layers(torch.cat([x, x3], dim=1))

    def _ms(self, x):
        x1 = self.ms1_layers(x)
        x2 = self.ms2_layers(x)
        x3 = self.ms3_layers(x)
        return self.ms4_layers(torch.cat([x, x1, x2, x3], dim=1))

    def forward(self, x):
        x1 = self._dsc(x)
        x2 = self._ms(x)
        x3 = self.final_layers(torch.cat([x1, x2], dim=1))
        return x3
