import os
from glob import glob
from logging import getLogger
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomCrop, Resize

logger = getLogger()


def minmax_scale(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    y = torch.clip(x, min=vmin, max=vmax)
    y = (y - vmin) / (vmax - vmin)
    return 2.0 * y - 1.0  # change value range [0,1] -> [-1, 1]


def minmax_scale_inverse(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    y = (x + 1.0) / 2.0
    return y * (vmax - vmin) + vmin


def robust_scale(x: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    return (x - mean) / scale


def robust_scale_inverse(x: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    return x * scale + mean


class VortexDatasetForBarotropicInstabilitySpectralNudging(Dataset):
    def __init__(
        self,
        data_dir: str,
        mean: float,
        std: float,
        lr_name: str,
        hr_name: str,
        image_width: int = None,
        image_height: int = None,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        dtype: torch.dtype = torch.float32,
        lr_method: str = None,
    ):
        self.mean, self.std = mean, std
        self.interpolation = interpolation
        self.dtype = dtype

        logger.info(f"LR = {lr_name}, HR = {hr_name}")
        self.lr_file_paths = sorted(glob(os.path.join(data_dir, lr_name, "*.npy")))
        self.hr_file_paths = sorted(glob(os.path.join(data_dir, hr_name, "*.npy")))

        assert len(self.lr_file_paths) == len(self.hr_file_paths)

        for p1, p2 in zip(self.lr_file_paths, self.hr_file_paths):
            assert os.path.basename(p1) == os.path.basename(p2)

        if image_height is None or image_width is None:
            self.image_size = None
            self.crop = None
            logger.info("Full image size is used")
        else:
            self.image_size = (image_height, image_width)
            self.crop = RandomCrop(self.image_size)
            logger.info(f"Image size {self.image_size} is used")

        if lr_method not in ["spectral_nudging", "average"]:
            raise Exception(f"{lr_method} is not supported!")
        self.lr_method = lr_method

        if self.lr_method == "average":
            logger.info(f"Local average is used to create LR data of {data_dir}")
            if lr_name == "T10" and hr_name == "T42":
                self.scale = 4
                self.unfold = torch.nn.Unfold(
                    kernel_size=(4, 4), stride=(4, 4), padding=(0, 0), dilation=(1, 1)
                )
            else:
                raise Exception("LR or HR name is not supported!")
        elif self.lr_method == "spectral_nudging":
            logger.info("Spectral nudged data are used for LR and HR pairs.")
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.lr_file_paths)

    def _read_numpy_data(self, path: str):
        data = torch.from_numpy(np.load(path)).to(self.dtype)
        return data

    def __getitem__(self, idx):
        hr_path = self.hr_file_paths[idx]
        hr_vortex = self._read_numpy_data(hr_path)
        hr_vortex = hr_vortex.unsqueeze(0)  # Add channel dim
        logger.debug(f"HR image size = {hr_vortex.shape}")

        if self.lr_method == "average":
            logger.debug("Average is used.")
            hr_vortex = hr_vortex[:, :, 1:]
            lr_vortex = self.unfold(hr_vortex.unsqueeze(0))  # add batch dim before unflod
            lr_vortex = torch.mean(lr_vortex, dim=1)  # average over sliding windows
            lr_vortex = lr_vortex.reshape(
                1, hr_vortex.shape[1] // self.scale, hr_vortex.shape[2] // self.scale
            )  # channel dim is 1
        else:
            lr_path = self.lr_file_paths[idx]
            lr_vortex = self._read_numpy_data(lr_path)
            lr_vortex = lr_vortex.unsqueeze(0)  # Add channel dim
        logger.debug(f"LR image size = {lr_vortex.shape}")

        lr_vortex = Resize(
            (hr_vortex.shape[1], hr_vortex.shape[2]), interpolation=self.interpolation
        )(lr_vortex)

        lr_vortex = robust_scale(lr_vortex, self.mean, self.std)
        hr_vortex = robust_scale(hr_vortex, self.mean, self.std)

        if self.crop is None:
            if self.lr_method == "average":
                return lr_vortex, hr_vortex
            else:
                return lr_vortex[:, :, 1:], hr_vortex[:, :, 1:]
            # Ignore points exactly on y boundary (channel wall).

        stacked = self.crop(torch.cat([lr_vortex, hr_vortex], dim=0))
        lr_vortex = stacked[0, :, :].unsqueeze(0)  # Add channel dim
        hr_vortex = stacked[1, :, :].unsqueeze(0)  # Add channel dim

        return lr_vortex, hr_vortex


class VortexDatasetForDecayingTurbulence(Dataset):
    def __init__(
        self,
        data_dir: str,
        mean: float,
        std: float,
        scale: int,
        image_width: int,
        image_height: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        dtype: torch.dtype = torch.float32,
        lr_method: str = None,
        num_simulations: int = None,
    ):
        self.mean, self.std = mean, std
        logger.info(f"Vorticity: mean = {self.mean}, std = {self.std}")

        self.scale = scale
        self.interpolation = interpolation
        self.dtype = dtype

        self.fit_image = True

        self.file_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        if num_simulations is not None:
            self.file_paths = self._extract_file_paths(data_dir, num_simulations)
        logger.info(f"Num of files = {len(self.file_paths)}")

        if image_height is None or image_width is None:
            self.image_size = None
            self.crop = None
        else:
            self.image_size = (image_height, image_width)
            logger.info(f"Image size = {self.image_size}")
            self.crop = RandomCrop(self.image_size)

        if lr_method not in ["average", "subsample"]:
            raise Exception(f"{lr_method} is not supported!")
        self.lr_method = lr_method

        if self.lr_method == "average":
            logger.info(f"Local average is used to create LR data of {data_dir}")
            unfold_size = (self.scale, self.scale)
            logger.info(f"Unfold_size = {unfold_size}")
            self.unfold = torch.nn.Unfold(
                kernel_size=unfold_size, stride=unfold_size, padding=(0, 0), dilation=(1, 1)
            )
        elif self.lr_method == "subsample":
            logger.info(f"Subsampling is used to create LR data of {data_dir}")

    def _extract_file_paths(self, data_dir: str, num_simulations: int) -> List[str]:
        file_paths = pd.Series(glob(os.path.join(data_dir, "*.npy")))

        simulation_names = file_paths.apply(
            lambda s: "_".join(os.path.basename(s).split("_")[:2])
        ).drop_duplicates()

        simulation_names = sorted(simulation_names.to_list())
        total_num = len(simulation_names)

        simulation_names = set(simulation_names[:num_simulations])
        actual_num = len(simulation_names)

        logger.info(f"Total simulation num = {total_num}, but only {actual_num} are used.")

        are_used = file_paths.apply(
            lambda s: "_".join(os.path.basename(s).split("_")[:2]) in simulation_names
        )
        file_paths = file_paths[are_used].to_list()

        logger.info(f"Extracted file num = {len(file_paths)}")

        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def _read_numpy_data(self, path: str):
        data = torch.from_numpy(np.load(path)).to(self.dtype)
        return data

    def __getitem__(self, idx):
        hr_vortex = self._read_numpy_data(self.file_paths[idx])
        hr_vortex = hr_vortex.unsqueeze(0)  # Add channel dim
        org_height, org_width = hr_vortex.shape[1], hr_vortex.shape[2]

        if self.fit_image:
            _h = int(np.round(hr_vortex.shape[1] / self.scale) * self.scale)  # height
            _w = int(np.round(hr_vortex.shape[2] / self.scale) * self.scale)  # width
            hr_vortex = Resize((_h, _w), interpolation=self.interpolation)(hr_vortex)

        if self.lr_method == "average":
            lr_vortex = self.unfold(hr_vortex.unsqueeze(0))  # add batch dim before unfold
            lr_vortex = torch.mean(lr_vortex, dim=1)  # average over each sliding window
            lr_vortex = lr_vortex.reshape(
                1, hr_vortex.shape[1] // self.scale, hr_vortex.shape[2] // self.scale
            )  # channel dim is 1
        elif self.lr_method == "subsample":

            lr_vortex = (
                hr_vortex[0, self.scale // 2 :: self.scale, self.scale // 2 :: self.scale]
                .detach()
                .clone()
            )
            lr_vortex = lr_vortex.unsqueeze(0)  # add channel dim

        lr_vortex = Resize(
            (hr_vortex.shape[1], hr_vortex.shape[2]), interpolation=self.interpolation
        )(lr_vortex)

        lr_vortex = robust_scale(lr_vortex, self.mean, self.std)
        hr_vortex = robust_scale(hr_vortex, self.mean, self.std)

        if self.fit_image:
            _size = (org_height, org_width)
            lr_vortex = Resize(_size, interpolation=self.interpolation)(lr_vortex)
            hr_vortex = Resize(_size, interpolation=self.interpolation)(hr_vortex)

        if self.crop is None:
            return lr_vortex, hr_vortex

        stacked = self.crop(torch.cat([lr_vortex, hr_vortex], dim=0))
        lr_vortex = stacked[0, :, :].unsqueeze(0)  # Add channel dim
        hr_vortex = stacked[1, :, :].unsqueeze(0)  # Add channel dim

        return lr_vortex, hr_vortex


class VortexDatasetForBarotropicInstability(Dataset):
    def __init__(
        self,
        data_dir: str,
        mean: float,
        std: float,
        scale: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        dtype: torch.dtype = torch.float32,
        lr_method: str = None,
    ):
        self.mean, self.std = mean, std
        logger.info(f"Vorticity: mean = {self.mean}, std = {self.std}")

        self.scale = scale
        self.interpolation = interpolation
        self.dtype = dtype

        self.fit_image = True

        self.file_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        logger.info(f"Data dir = {data_dir}")
        logger.info(f"Num of files = {len(self.file_paths)}")

        if lr_method not in ["average", "subsample"]:
            raise Exception(f"{lr_method} is not supported!")
        self.lr_method = lr_method

        if self.lr_method == "average":
            logger.info("Local average is used to create LR data")
            unfold_size = (self.scale, self.scale)
            logger.info(f"Unfold_size = {unfold_size}")
            self.unfold = torch.nn.Unfold(
                kernel_size=unfold_size, stride=unfold_size, padding=(0, 0), dilation=(1, 1)
            )
        elif self.lr_method == "subsample":
            logger.info("Subsampling is used to create LR data")

    def __len__(self):
        return len(self.file_paths)

    def _read_numpy_data(self, path: str):
        data = torch.from_numpy(np.load(path)).to(self.dtype)
        return data

    def __getitem__(self, idx):
        hr_vortex = self._read_numpy_data(self.file_paths[idx])
        hr_vortex = hr_vortex.unsqueeze(0)  # Add channel dim
        org_height, org_width = hr_vortex.shape[1], hr_vortex.shape[2]
        logger.debug(f"Org hr image size = {hr_vortex.shape}")

        if self.fit_image:
            _h = int(np.round(hr_vortex.shape[1] / self.scale) * self.scale)  # height
            _w = int(np.round(hr_vortex.shape[2] / self.scale) * self.scale)  # width
            hr_vortex = Resize((_h, _w), interpolation=self.interpolation)(hr_vortex)
            logger.debug(f"Fitted hr image size = {hr_vortex.shape}")

        if self.lr_method == "average":
            lr_vortex = self.unfold(hr_vortex.unsqueeze(0))  # add batch dim before unfold
            lr_vortex = torch.mean(lr_vortex, dim=1)  # average over each sliding window
            lr_vortex = lr_vortex.reshape(
                1, hr_vortex.shape[1] // self.scale, hr_vortex.shape[2] // self.scale
            )  # channel dim is 1
        elif self.lr_method == "subsample":
            lr_vortex = (
                hr_vortex[0, self.scale // 2 :: self.scale, self.scale // 2 :: self.scale]
                .detach()
                .clone()
            )
            lr_vortex = lr_vortex.unsqueeze(0)  # add channel dim

        lr_vortex = Resize(
            (hr_vortex.shape[1], hr_vortex.shape[2]), interpolation=self.interpolation
        )(lr_vortex)

        lr_vortex = robust_scale(lr_vortex, self.mean, self.std)
        hr_vortex = robust_scale(hr_vortex, self.mean, self.std)

        if self.fit_image:
            _size = (org_height, org_width)
            lr_vortex = Resize(_size, interpolation=self.interpolation)(lr_vortex)
            hr_vortex = Resize(_size, interpolation=self.interpolation)(hr_vortex)

        return lr_vortex[:, :, 1:], hr_vortex[:, :, 1:]
        # Ignore grid points on y boundary
