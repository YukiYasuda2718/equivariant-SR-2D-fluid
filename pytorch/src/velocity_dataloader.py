import os
from logging import getLogger
from typing import Tuple

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import get_torch_generator, seed_worker

from velocity_dataset import (
    VelocityDatasetForBarotropicInstability,
    VelocityDatasetForBarotropicInstabilitySpectralNudging,
    VelocityDatasetForDecayingTurbulence,
)

logger = getLogger()


def make_velocity_dataloaders_for_barotropic_instability_spectral_nudging(
    data_dir: str, config: dict, num_workers: int = 2, seed: int = 0
) -> dict:
    dict_dataloaders = {}
    for kind in ["train", "valid", "test"]:
        dataset = VelocityDatasetForBarotropicInstabilitySpectralNudging(
            data_dir=os.path.join(data_dir, kind),
            u_mean=config["data"]["u_mean"],
            u_std=config["data"]["u_std"],
            v_mean=config["data"]["v_mean"],
            v_std=config["data"]["v_std"],
            lr_name=config["data"]["LR_name"],
            hr_name=config["data"]["HR_name"],
            image_width=config["data"].get("image_width", None),
            image_height=config["data"].get("image_height", None),
            lr_method=config["data"]["creation_method"],
        )

        dict_dataloaders[kind] = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            drop_last=True if kind == "train" else False,
            shuffle=True if kind == "train" else False,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=get_torch_generator(seed),
        )
        logger.info(
            f"{kind} dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
        )

    return dict_dataloaders


def make_velocity_dataloaders_for_decaying_turbulence(
    data_dir: str, config: dict, num_workers: int = 2, seed: int = 0
) -> dict:
    dict_dataloaders = {}
    for kind in ["train", "valid", "test"]:
        dataset = VelocityDatasetForDecayingTurbulence(
            os.path.join(data_dir, kind),
            mean=config["data"]["velocity_mean"],
            std=config["data"]["velocity_std"],
            scale=config["data"]["scale"],
            image_width=config["data"]["image_width"] if kind == "train" else None,
            image_height=config["data"]["image_height"] if kind == "train" else None,
            lr_method=config["data"]["creation_method"],
            num_simulations=config["data"].get(f"num_simulations_{kind}", None),
        )

        dict_dataloaders[kind] = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            drop_last=True if kind == "train" else False,
            shuffle=True if kind == "train" else False,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=get_torch_generator(seed),
        )
        logger.info(
            f"{kind} dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
        )

    return dict_dataloaders


def make_velocity_distributed_dataloaders_for_decaying_turbulence(
    rank: int,
    world_size: int,
    data_dir: str,
    config: dict,
    num_workers: int = 2,
    seed: int = 0,
) -> Tuple[dict, dict]:
    dict_dataloaders, dict_samplers = {}, {}
    for kind in ["train", "valid", "test"]:
        dataset = VelocityDatasetForDecayingTurbulence(
            os.path.join(data_dir, kind),
            mean=config["data"]["velocity_mean"],
            std=config["data"]["velocity_std"],
            scale=config["data"]["scale"],
            image_width=config["data"]["image_width"] if kind == "train" else None,
            image_height=config["data"]["image_height"] if kind == "train" else None,
            lr_method=config["data"]["creation_method"],
            num_simulations=config["data"].get(f"num_simulations_{kind}", None),
        )

        dict_samplers[kind] = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=True if kind == "train" else False,
            drop_last=True if kind == "train" else False,
        )

        dict_dataloaders[kind] = DataLoader(
            dataset,
            sampler=dict_samplers[kind],
            batch_size=config["train"]["batch_size"] // world_size,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=get_torch_generator(seed),
        )

        if rank == 0:
            logger.info(
                f"{kind} dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
            )

    return dict_dataloaders, dict_samplers


def make_velocity_dataloaders_for_barotropic_instability(
    data_dir: str, config: dict, num_workers: int = 2, seed: int = 0
) -> dict:
    dict_dataloaders = {}
    for kind in ["train", "valid", "test"]:
        dataset = VelocityDatasetForBarotropicInstability(
            data_dir=os.path.join(data_dir, kind),
            u_mean=config["data"]["u_mean"],
            u_std=config["data"]["u_std"],
            v_mean=config["data"]["v_mean"],
            v_std=config["data"]["v_std"],
            scale=config["data"]["scale"],
            lr_method=config["data"]["creation_method"],
            num_simulations=config["data"].get(f"num_simulations_{kind}", None),
        )

        dict_dataloaders[kind] = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            drop_last=True if kind == "train" else False,
            shuffle=True if kind == "train" else False,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=get_torch_generator(seed),
        )
        logger.info(
            f"{kind} dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
        )

    return dict_dataloaders
