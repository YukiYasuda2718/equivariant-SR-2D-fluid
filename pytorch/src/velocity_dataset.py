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


class VelocityDatasetForBarotropicInstabilitySpectralNudging(Dataset):
    def __init__(
        self,
        data_dir: str,
        u_mean: float,
        u_std: float,
        v_mean: float,
        v_std: float,
        lr_name: str,
        hr_name: str,
        image_width: int = None,
        image_height: int = None,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        dtype: torch.dtype = torch.float32,
        lr_method: str = None,
    ):
        self.u_mean, self.u_std = u_mean, u_std
        self.v_mean, self.v_std = v_mean, v_std
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
            raise NotImplementedError("Cropping is not implemented yet.")

        if lr_method not in ["spectral_nudging", "average"]:
            raise Exception(f"{lr_method} is not supported!")
        self.lr_method = lr_method

        if self.lr_method == "spectral_nudging":
            logger.info("Spectral nudged data are used for LR and HR pairs.")
            if lr_name == "T10" and hr_name == "T42":
                self.scale = 4
            else:
                raise Exception("LR or HR name is not supported!")
        elif self.lr_method == "average":
            logger.info(f"Local average is used to create LR data of {data_dir}")
            if lr_name == "T10" and hr_name == "T42":
                self.scale = 4
                self.unfold = torch.nn.Unfold(
                    kernel_size=(4, 4), stride=(4, 4), padding=(0, 0), dilation=(1, 1)
                )
            else:
                raise Exception("LR or HR name is not supported!")
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.lr_file_paths)

    def _read_numpy_data(self, path: str):
        data = torch.from_numpy(np.load(path)).to(self.dtype)
        return data

    def _average(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        lr_data = self.unfold(hr_data.unsqueeze(0))  # add batch dim before unfold
        lr_data = torch.mean(lr_data, dim=1)  # average over each sliding window
        return lr_data.reshape(1, hr_data.shape[1] // self.scale, hr_data.shape[2] // self.scale)

    def _fit(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        _h = int(np.round(hr_data.shape[1] / self.scale) * self.scale)  # height
        _w = int(np.round(hr_data.shape[2] / self.scale) * self.scale)  # width
        return Resize((_h, _w), interpolation=self.interpolation)(hr_data)

    def __getitem__(self, idx):
        hr_path = self.hr_file_paths[idx]
        hr_velocity = self._read_numpy_data(hr_path)
        logger.debug(f"HR image size = {hr_velocity.shape}")
        _u = self._fit(hr_velocity[0, :, :].unsqueeze(0))
        _v = self._fit(hr_velocity[1, :, :].unsqueeze(0))
        hr_velocity = torch.cat([_u, _v])
        logger.debug(f"HR image size = {hr_velocity.shape}")

        if self.lr_method == "average":
            logger.debug("Average is used")
            _u = self._average(hr_velocity[0, :, :].unsqueeze(0))  # add channel dim
            _v = self._average(hr_velocity[1, :, :].unsqueeze(0))
            lr_velocity = torch.cat([_u, _v])
        else:
            lr_path = self.lr_file_paths[idx]
            lr_velocity = self._read_numpy_data(lr_path)
            logger.debug(f"LR image size = {lr_velocity.shape}")
            _u = self._fit(lr_velocity[0, :, :].unsqueeze(0))
            _v = self._fit(lr_velocity[1, :, :].unsqueeze(0))
            lr_velocity = torch.cat([_u, _v])
        logger.debug(f"LR image size = {lr_velocity.shape}")

        lr_velocity = Resize(
            (hr_velocity.shape[1], hr_velocity.shape[2]), interpolation=self.interpolation
        )(lr_velocity)

        lr_velocity[0, :, :] = robust_scale(lr_velocity[0, :, :], self.u_mean, self.u_std)
        hr_velocity[0, :, :] = robust_scale(hr_velocity[0, :, :], self.u_mean, self.u_std)
        lr_velocity[1, :, :] = robust_scale(lr_velocity[1, :, :], self.v_mean, self.v_std)
        hr_velocity[1, :, :] = robust_scale(hr_velocity[1, :, :], self.v_mean, self.v_std)

        if self.crop is None:
            return lr_velocity, hr_velocity
        else:
            raise NotImplementedError()


class VelocityDatasetForDecayingTurbulence(Dataset):
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
        logger.info(f"Velocity: mean = {self.mean}, std = {self.std}")

        self.scale = scale
        self.interpolation = interpolation
        self.dtype = dtype

        self.fit_image = False
        if self.scale % 2 == 1:
            self.fit_image = True
            logger.info(f"Scale = {self.scale} (odd number), so fit HR images each time.")
        else:
            if self.scale % 4 == 0:
                pass
            elif self.scale % 8 == 0:
                pass
            else:
                raise NotImplementedError()

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

    def _average(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        lr_data = self.unfold(hr_data.unsqueeze(0))  # add batch dim before unfold
        lr_data = torch.mean(lr_data, dim=1)  # average over each sliding window
        lr_data = lr_data.reshape(
            1, hr_data.shape[1] // self.scale, hr_data.shape[2] // self.scale
        )  # channel dim is 1
        return Resize((hr_data.shape[1], hr_data.shape[2]), interpolation=self.interpolation)(
            lr_data
        )

    def _fit(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        _h = int(np.round(hr_data.shape[1] / self.scale) * self.scale)  # height
        _w = int(np.round(hr_data.shape[2] / self.scale) * self.scale)  # width
        return Resize((_h, _w), interpolation=self.interpolation)(hr_data)

    def _subsample(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        lr_data = (
            hr_data[0, self.scale // 2 :: self.scale, self.scale // 2 :: self.scale]
            .detach()
            .clone()
        )
        lr_data = lr_data.unsqueeze(0)  # add channel dim
        return Resize((hr_data.shape[1], hr_data.shape[2]), interpolation=self.interpolation)(
            lr_data
        )

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
        velocity = self._read_numpy_data(self.file_paths[idx])
        # Separate components and add channel dims
        hr_u, hr_v = velocity[0, :, :].unsqueeze(0), velocity[1, :, :].unsqueeze(0)
        org_height, org_width = hr_u.shape[1], hr_u.shape[2]

        if self.fit_image:
            hr_u = self._fit(hr_u)
            hr_v = self._fit(hr_v)

        if self.lr_method == "average":
            lr_u = self._average(hr_u)
            lr_v = self._average(hr_v)
        elif self.lr_method == "subsample":
            lr_u = self._subsample(hr_u)
            lr_v = self._subsample(hr_v)

        lr_u = robust_scale(lr_u, self.mean, self.std)
        lr_v = robust_scale(lr_v, self.mean, self.std)
        hr_u = robust_scale(hr_u, self.mean, self.std)
        hr_v = robust_scale(hr_v, self.mean, self.std)

        if self.fit_image:
            _size = (org_height, org_width)
            hr_u = Resize(_size, interpolation=self.interpolation)(hr_u)
            hr_v = Resize(_size, interpolation=self.interpolation)(hr_v)
            lr_u = Resize(_size, interpolation=self.interpolation)(lr_u)
            lr_v = Resize(_size, interpolation=self.interpolation)(lr_v)

        hr_velocity = torch.cat([hr_u, hr_v])
        lr_velocity = torch.cat([lr_u, lr_v])

        assert hr_velocity.shape[0] == 2
        assert lr_velocity.shape[0] == 2
        assert len(hr_velocity.shape) == 3
        assert len(lr_velocity.shape) == 3

        if self.crop is None:
            return lr_velocity, hr_velocity

        stacked = self.crop(torch.cat([lr_velocity, hr_velocity], dim=0))
        lr_velocity = stacked[0:2, :, :]
        hr_velocity = stacked[2:4, :, :]

        assert hr_velocity.shape[0] == 2
        assert lr_velocity.shape[0] == 2
        assert len(hr_velocity.shape) == 3
        assert len(lr_velocity.shape) == 3

        return lr_velocity, hr_velocity


class VelocityDatasetForBarotropicInstability(Dataset):
    def __init__(
        self,
        data_dir: str,
        u_mean: float,
        u_std: float,
        v_mean: float,
        v_std: float,
        scale: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        dtype: torch.dtype = torch.float32,
        lr_method: str = None,
    ):
        self.u_mean, self.u_std = u_mean, u_std
        self.v_mean, self.v_std = v_mean, v_std
        logger.info(f"U: mean = {self.u_mean}, std = {self.u_std}")
        logger.info(f"V: mean = {self.v_mean}, std = {self.v_std}")

        self.scale = scale
        self.interpolation = interpolation
        self.dtype = dtype

        self.fit_image = False
        if self.scale % 2 == 1:
            self.fit_image = True
            logger.info(f"Scale = {self.scale} (odd number), so fit HR images each time.")
        else:
            raise Exception("Not tested yet.")

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

    def _average(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        lr_data = self.unfold(hr_data.unsqueeze(0))  # add batch dim before unfold
        lr_data = torch.mean(lr_data, dim=1)  # average over each sliding window
        lr_data = lr_data.reshape(
            1, hr_data.shape[1] // self.scale, hr_data.shape[2] // self.scale
        )  # channel dim is 1
        return Resize((hr_data.shape[1], hr_data.shape[2]), interpolation=self.interpolation)(
            lr_data
        )

    def _fit(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        _h = int(np.round(hr_data.shape[1] / self.scale) * self.scale)  # height
        _w = int(np.round(hr_data.shape[2] / self.scale) * self.scale)  # width
        return Resize((_h, _w), interpolation=self.interpolation)(hr_data)

    def _subsample(self, hr_data):
        assert len(hr_data.shape) == 3
        assert hr_data.shape[0] == 1  # channel num is 1
        lr_data = (
            hr_data[0, self.scale // 2 :: self.scale, self.scale // 2 :: self.scale]
            .detach()
            .clone()
        )
        lr_data = lr_data.unsqueeze(0)  # add channel dim
        return Resize((hr_data.shape[1], hr_data.shape[2]), interpolation=self.interpolation)(
            lr_data
        )

    def __len__(self):
        return len(self.file_paths)

    def _read_numpy_data(self, path: str):
        data = torch.from_numpy(np.load(path)).to(self.dtype)
        return data

    def __getitem__(self, idx):
        velocity = self._read_numpy_data(self.file_paths[idx])
        logger.debug(f"Org hr image size = {velocity.shape}")

        # Separate components and add channel dims
        hr_u, hr_v = velocity[0, :, :].unsqueeze(0), velocity[1, :, :].unsqueeze(0)
        org_height, org_width = hr_u.shape[1], hr_u.shape[2]

        if self.fit_image:
            hr_u = self._fit(hr_u)
            hr_v = self._fit(hr_v)
            logger.debug(f"Fitted U size = {hr_u.shape}, V size = {hr_v.shape}")

        if self.lr_method == "average":
            lr_u = self._average(hr_u)
            lr_v = self._average(hr_v)
        elif self.lr_method == "subsample":
            lr_u = self._subsample(hr_u)
            lr_v = self._subsample(hr_v)

        lr_u = robust_scale(lr_u, self.u_mean, self.u_std)
        lr_v = robust_scale(lr_v, self.v_mean, self.v_std)
        hr_u = robust_scale(hr_u, self.u_mean, self.u_std)
        hr_v = robust_scale(hr_v, self.v_mean, self.v_std)

        if self.fit_image:
            _size = (org_height, org_width)
            hr_u = Resize(_size, interpolation=self.interpolation)(hr_u)
            hr_v = Resize(_size, interpolation=self.interpolation)(hr_v)
            lr_u = Resize(_size, interpolation=self.interpolation)(lr_u)
            lr_v = Resize(_size, interpolation=self.interpolation)(lr_v)

        hr_velocity = torch.cat([hr_u, hr_v])
        lr_velocity = torch.cat([lr_u, lr_v])

        assert hr_velocity.shape[0] == 2
        assert lr_velocity.shape[0] == 2
        assert len(hr_velocity.shape) == 3
        assert len(lr_velocity.shape) == 3

        return lr_velocity[:, :, 1:], hr_velocity[:, :, 1:]
        # Ignore grid points on y boundary
