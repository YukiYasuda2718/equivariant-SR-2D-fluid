from typing import Tuple

import numpy as np

ENDIAN = "<"  # Little endia}n
NX_T10, NY_T10 = 32, 16
NX_T21, NY_T21 = 64, 32
NX_T42, NY_T42 = 128, 64
NX_T85, NY_T85 = 256, 128


def read_simulation_results(
    data_dir: str, config_name: str, seed: int, use_T85_data: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    with open(f"{data_dir}/{config_name}/vortex_grid_T10_{config_name}_seed{seed}.dat", "r") as f:
        vortex_field_T10 = np.fromfile(f, f"{ENDIAN}d", -1).reshape(-1, NX_T10, NY_T10 + 1)

    with open(f"{data_dir}/{config_name}/vortex_grid_T21_{config_name}_seed{seed}.dat", "r") as f:
        vortex_field_T21 = np.fromfile(f, f"{ENDIAN}d", -1).reshape(-1, NX_T21, NY_T21 + 1)

    with open(f"{data_dir}/{config_name}/vortex_grid_T42_{config_name}_seed{seed}.dat", "r") as f:
        vortex_field_T42 = np.fromfile(f, f"{ENDIAN}d", -1).reshape(-1, NX_T42, NY_T42 + 1)

    vortex_field_T85 = np.array([])
    if use_T85_data:
        with open(
            f"{data_dir}/{config_name}/vortex_grid_T85_{config_name}_seed{seed}.dat", "r"
        ) as f:
            vortex_field_T85 = np.fromfile(f, f"{ENDIAN}d", -1).reshape(-1, NX_T85, NY_T85 + 1)

    return vortex_field_T10, vortex_field_T21, vortex_field_T42, vortex_field_T85


def read_vortex_file(file_path: str, nx: int, ny: int):
    with open(file_path, "r") as f:
        return np.fromfile(f, f"{ENDIAN}d", -1).reshape(-1, nx, ny)


def read_simulation_result_for_T42_resolution(data_dir: str, config_name: str, seed: int):
    with open(f"{data_dir}/{config_name}/vortex_grid_T42_{config_name}_seed{seed}.dat", "r") as f:
        vortex_field_T42 = np.fromfile(f, f"{ENDIAN}d", -1).reshape(-1, NX_T42, NY_T42 + 1)
    return vortex_field_T42
