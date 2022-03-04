from logging import getLogger

from e2cnn import gspaces
from e2cnn import nn as enn

from dscms import DSCMS
from rrdn import RRDN
from eq_rrdn import EqRRDN
from eq_dscms import EqDSCMS

logger = getLogger()


def make_model(config: dict):
    if config["model"]["name"] == "Eq-DSC-MS":
        model = make_EqDSCMS(config)
    elif config["model"]["name"] == "Eq-RRDN":
        model = make_EqRRDN(config)
    elif config["model"]["name"] == "DSC-MS":
        model = DSCMS(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            factor_filter_num=config["model"]["factor_filter_num"],
        )
    elif config["model"]["name"] == "RRDN":
        model = RRDN(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            num_features=config["model"]["num_features"],
            post_features=config["model"]["num_post_features"],
            growth_rate=config["model"]["growth_rates"],
            num_layers=config["model"]["num_layers"],
            kernel_size=config["model"]["kernel_size"],
            num_blocks=config["model"]["num_blocks"],
        )
    else:
        logger.error(f'Model {config["model"]["name"]} is not supported')
        raise Exception(f'Model {config["model"]["name"]} is not supported')
    return model
