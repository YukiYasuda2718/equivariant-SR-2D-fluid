from logging import getLogger

from dscms import DSCMS
from e2cnn import gspaces
from e2cnn import nn as enn
from eq_dscms import EqDSCMS
from eq_rrdn import EqRRDN
from rrdn import RRDN

logger = getLogger()


def make_EqDSCMS(config: dict):
    R2_ACT = gspaces.Rot2dOnR2(N=config["model"]["degree_rotation"])
    logger.info(f'Degree of rotation = {config["model"]["degree_rotation"]}')

    def get_field_type(str_type: str):
        type_name, type_num = str_type.split(",")[0], int(str_type.split(",")[1])
        logger.info(f"e2eq: type_name = {type_name}, type_num = {type_num}")

        if type_name == "trivial_repr":
            return enn.FieldType(R2_ACT, type_num * [R2_ACT.trivial_repr])  # scalar
        elif type_name == "regular_repr":
            return enn.FieldType(R2_ACT, type_num * [R2_ACT.regular_repr])  # regular representation
        elif type_name == "irrep(1)":
            return enn.FieldType(R2_ACT, type_num * [R2_ACT.irrep(1)])  # irreducible representation
        else:
            raise NotImplementedError()

    if config["model"]["in_channels"] == 1 and config["model"]["out_channels"] == 1:
        in_type = enn.FieldType(R2_ACT, [R2_ACT.trivial_repr])  # scalar
        out_type = enn.FieldType(R2_ACT, [R2_ACT.trivial_repr])
        logger.info("In and out types are scalar")
    elif config["model"]["in_channels"] == 2 and config["model"]["out_channels"] == 2:
        if config["model"]["degree_rotation"] == 2:
            in_type = enn.FieldType(R2_ACT, 2 * [R2_ACT.irrep(1)])
            out_type = enn.FieldType(R2_ACT, 2 * [R2_ACT.irrep(1)])
        else:
            in_type = enn.FieldType(R2_ACT, [R2_ACT.irrep(1)])
            out_type = enn.FieldType(R2_ACT, [R2_ACT.irrep(1)])
        logger.info("In and out types are vector")
    else:
        raise NotImplementedError()

    return EqDSCMS(
        in_type=in_type,
        out_type=out_type,
        dsc_feature_type=get_field_type(config["model"]["dsc_feature_type"]),
        ms1_feature_type=get_field_type(config["model"]["ms1_feature_type"]),
        ms2_feature_type=get_field_type(config["model"]["ms2_feature_type"]),
        ms3_feature_type=get_field_type(config["model"]["ms3_feature_type"]),
        ms4_feature_type=get_field_type(config["model"]["ms4_feature_type"]),
    )


def make_EqRRDN(config: dict):
    R2_ACT = gspaces.Rot2dOnR2(N=config["model"]["degree_rotation"])
    logger.info(f'Degree of rotation = {config["model"]["degree_rotation"]}')

    def get_field_type(str_type: str):
        type_name, type_num = str_type.split(",")[0], int(str_type.split(",")[1])
        logger.info(f"e2eq: type_name = {type_name}, type_num = {type_num}")

        if type_name == "trivial_repr":
            return enn.FieldType(R2_ACT, type_num * [R2_ACT.trivial_repr])  # scalar
        elif type_name == "regular_repr":
            return enn.FieldType(R2_ACT, type_num * [R2_ACT.regular_repr])  # regular representation
        elif type_name == "irrep(1)":
            return enn.FieldType(R2_ACT, type_num * [R2_ACT.irrep(1)])  # irreducible representation
        else:
            raise NotImplementedError()

    if config["model"]["in_channels"] == 1 and config["model"]["out_channels"] == 1:
        in_type = enn.FieldType(R2_ACT, [R2_ACT.trivial_repr])  # scalar
        out_type = enn.FieldType(R2_ACT, [R2_ACT.trivial_repr])
        logger.info("In and out types are scalar")
    elif config["model"]["in_channels"] == 2 and config["model"]["out_channels"] == 2:
        if config["model"]["degree_rotation"] == 2:
            in_type = enn.FieldType(R2_ACT, 2 * [R2_ACT.irrep(1)])
            out_type = enn.FieldType(R2_ACT, 2 * [R2_ACT.irrep(1)])
        else:
            in_type = enn.FieldType(R2_ACT, [R2_ACT.irrep(1)])
            out_type = enn.FieldType(R2_ACT, [R2_ACT.irrep(1)])
        logger.info("In and out types are vector")
    else:
        raise NotImplementedError()

    return EqRRDN(
        in_type=in_type,
        out_type=out_type,
        feature_type=get_field_type(config["model"]["feature_type"]),
        post_feature_type=get_field_type(config["model"]["post_feature_type"]),
        growth_type=get_field_type(config["model"]["growth_type"]),
        kernel_size=config["model"]["kernel_size"],
        num_layers=config["model"]["num_layers"],
        num_blocks=config["model"]["num_blocks"],
    )


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
