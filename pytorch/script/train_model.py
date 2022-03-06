import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from logging import INFO, FileHandler, StreamHandler, getLogger
from time import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = "/workspace"
sys.path.append(f"{ROOT_DIR}/pytorch/src")
sys.path.append(f"/{ROOT_DIR}/pytorch/model")

from model_maker import make_model
from trainer import train
from utils import set_seeds
from validator import validate
from velocity_dataloader import (
    make_velocity_dataloaders_for_barotropic_instability,
    make_velocity_dataloaders_for_barotropic_instability_spectral_nudging,
    make_velocity_dataloaders_for_decaying_turbulence,
)
from vortex_dataloader import (
    make_vortex_dataloaders_for_barotropic_instability,
    make_vortex_dataloaders_for_barotropic_instability_spectral_nudging,
    make_vortex_dataloaders_for_decaying_turbulence,
)

logger = getLogger()
log_handler = StreamHandler(sys.stdout)
logger.addHandler(log_handler)
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, required=True)
parser.add_argument("--cuda_num", type=int, default=None)
parser.add_argument("--data_method", type=str, required=True)
parser.add_argument("--experiment_name", type=str, required=True)


def get_dataloaders(experiment_name: str, data_kind: str, data_dir: str, config: dict) -> dict:
    if experiment_name == "decaying_turbulence":
        if data_kind == "vorticity":
            return make_vortex_dataloaders_for_decaying_turbulence(data_dir, config)
        else:
            return make_velocity_dataloaders_for_decaying_turbulence(data_dir, config)
    elif experiment_name == "barotropic_instability":
        if data_kind == "vorticity":
            return make_vortex_dataloaders_for_barotropic_instability(data_dir, config)
        else:
            return make_velocity_dataloaders_for_barotropic_instability(data_dir, config)
    elif experiment_name == "barotropic_instability_spectral_nudging":
        if data_kind == "vorticity":
            return make_vortex_dataloaders_for_barotropic_instability_spectral_nudging(
                data_dir, config
            )
        else:
            return make_velocity_dataloaders_for_barotropic_instability_spectral_nudging(
                data_dir, config
            )
    else:
        logger.error(f"{experiment_name} is not supported")
        raise Exception(f"{experiment_name} is not supported")


if __name__ == "__main__":
    NOW = datetime.strftime(datetime.now(), "%Y%m%dT%H%M%S")
    CONFIG_NAME = parser.parse_args().config_name
    CUDA_NUM = parser.parse_args().cuda_num
    DATA_METHOD = parser.parse_args().data_method
    EXPERIMENT_NAME = parser.parse_args().experiment_name

    # Read config
    CONFIG_PATH = f"{ROOT_DIR}/pytorch/config/{EXPERIMENT_NAME}/{DATA_METHOD}/{CONFIG_NAME}.yml"
    with open(CONFIG_PATH) as file:
        CONFIG = yaml.safe_load(file)
    DATA_KIND = "vorticity" if CONFIG["model"]["in_channels"] == 1 else "velocity"
    assert CONFIG["data"]["creation_method"] == DATA_METHOD
    assert CONFIG["fortran"]["experiment_name"] == EXPERIMENT_NAME

    if CONFIG["model"]["name"].startswith("Eq"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"

    # Set paths
    TENSOR_BOARD_LOG_DIR = f'{ROOT_DIR}/log/pytorch/{CONFIG["fortran"]["experiment_name"]}/{CONFIG["data"]["creation_method"]}/{CONFIG["model"]["name"]}/{CONFIG_NAME}/{NOW}'
    DATA_DIR = f'{ROOT_DIR}/data/pytorch/{CONFIG["fortran"]["experiment_name"]}/DL_data/{DATA_KIND}'
    OUTPUT_DIR = f'{ROOT_DIR}/data/pytorch/{CONFIG["fortran"]["experiment_name"]}/DL_results/{CONFIG["data"]["creation_method"]}/{CONFIG["model"]["name"]}/{CONFIG_NAME}'

    os.makedirs(TENSOR_BOARD_LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_WEIGHT_PATH = f"{OUTPUT_DIR}/weights.pth"
    LEARNING_CURVE_PATH = f"{OUTPUT_DIR}/learning_curves_{NOW}.csv"
    logger.addHandler(FileHandler(f"{OUTPUT_DIR}/train_{NOW}.log"))

    if os.path.exists(MODEL_WEIGHT_PATH):
        logger.error("Training results already exist!!")
        raise Exception("Training results already exist!!")

    logger.info(f"EXPERIMENT_NAME = {EXPERIMENT_NAME}")
    logger.info(f"DATA_METHOD = {DATA_METHOD}")
    logger.info(f"CONFIG_PATH = {CONFIG_PATH}")
    logger.info(f"Data kind = {DATA_KIND}")
    logger.info(f"TENSOR_BOARD_LOG_DIR = {TENSOR_BOARD_LOG_DIR}")
    logger.info(f"DATA_DIR = {DATA_DIR}")
    logger.info(f"OUTPUT_DIR = {OUTPUT_DIR}")
    logger.info(f"MODEL_WEIGHT_PATH = {MODEL_WEIGHT_PATH}")
    logger.info(f"LEARNING_CURVE_PATH = {LEARNING_CURVE_PATH}")

    # Set DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        logger.info("GPU is used.")
        if CUDA_NUM is not None:
            DEVICE = f"{DEVICE}:{CUDA_NUM}"
            logger.info(f"CUDA = {DEVICE}")
    else:
        logger.error("No GPU!! CPU is used.")
        raise Exception("No GPU!! CPU is used.")

    # Make dataloaders and model
    set_seeds(CONFIG["train"]["seed"])
    dict_dataloaders = get_dataloaders(EXPERIMENT_NAME, DATA_KIND, DATA_DIR, CONFIG)

    model = make_model(CONFIG)
    model = model.to(DEVICE)

    logger.info("Loss is MSELoss.")
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["train"]["lr"])
    logger.info(f'Adam Optimizer learning rate = {CONFIG["train"]["lr"]}')

    # Train model
    early_stop_count = 0
    best_weights = deepcopy(model.state_dict())
    best_epoch = 0
    best_loss, best_psnr = np.inf, -np.inf
    all_scores = []

    writer = SummaryWriter(log_dir=TENSOR_BOARD_LOG_DIR)

    start_time = time()
    logger.info(f"Train start: {datetime.now().isoformat()}")

    for epoch in range(CONFIG["train"]["num_epochs"]):
        _time = time()
        logger.info(f'Epoch: {epoch + 1} / {CONFIG["train"]["num_epochs"]}')

        loss, _, psnr = train(
            dict_dataloaders["train"],
            model,
            loss_fn,
            optimizer,
            DEVICE,
            tqdm_message=f'epoch: {epoch}/{CONFIG["train"]["num_epochs"]-1}',
        )
        val_loss, _, val_psnr = validate(dict_dataloaders["valid"], model, loss_fn, DEVICE)

        scores = {
            "loss": loss,
            "val_loss": val_loss,
            "psnr": psnr,
            "val_psnr": val_psnr,
        }
        all_scores.append(scores)
        writer.add_scalars("loss", {k: v for k, v in scores.items() if "loss" in k}, epoch)
        writer.add_scalars("psnr", {k: v for k, v in scores.items() if "psnr" in k}, epoch)

        if val_loss <= best_loss:
            logger.info(f"Best loss is updated: {best_loss:.8f} -> {val_loss:.8f}")
            best_epoch = epoch
            best_loss, best_psnr = val_loss, val_psnr
            best_weights = deepcopy(model.state_dict())

            torch.save(best_weights, MODEL_WEIGHT_PATH)
            df_scores = pd.DataFrame(all_scores)
            df_scores.to_csv(LEARNING_CURVE_PATH, index=False)
            early_stop_count = 0
        else:
            early_stop_count += 1
            logger.info(f"Early stopping count = {early_stop_count}")

            if epoch % 10 == 0:
                df_scores = pd.DataFrame(all_scores)
                df_scores.to_csv(LEARNING_CURVE_PATH, index=False)

        logger.info(f"Elapsed time = {time() - _time} sec")
        logger.info("-----")

        if early_stop_count >= CONFIG["train"]["early_stopping_patience"]:
            logger.info(
                f'Early stopped. count =  {early_stop_count} / {CONFIG["train"]["early_stopping_patience"]}'
            )
            break

    logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}, best_psnr: {best_psnr:.4f}")
    torch.save(best_weights, MODEL_WEIGHT_PATH)

    writer.close()
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv(LEARNING_CURVE_PATH, index=False)

    end_time = time()
    logger.info(f"Train end: {datetime.now().isoformat()}")
    logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")
