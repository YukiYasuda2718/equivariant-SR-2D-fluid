from typing import Tuple

import torch
import torch.nn.functional as F
from e2cnn.gspaces import GeneralOnR2
from e2cnn.nn import EquivariantModule, FieldType, GeometricTensor


class LeakyReLU(EquivariantModule):
    """This is a wrapper of LeakyReLU fitting into e2cnn EquivariantModule."""

    def __init__(self, in_type: FieldType, negative_slope: float = 0.01, inplace: bool = False):
        assert isinstance(in_type.gspace, GeneralOnR2)

        super(LeakyReLU, self).__init__()

        for r in in_type.representations:
            assert (
                "pointwise" in r.supported_nonlinearities
            ), 'Error! Representation "{}" does not support "pointwise" non-linearity'.format(
                r.name
            )

        self.space = in_type.gspace
        self.in_type = in_type

        # the representation in input is preserved
        self.out_type = in_type

        self._inplace = inplace
        self._negative_slope = negative_slope

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert (
            input.type == self.in_type
        ), "Error! the type of the input does not match the input type of this module"
        return GeometricTensor(
            F.leaky_relu(input.tensor, negative_slope=self._negative_slope, inplace=self._inplace),
            self.out_type,
        )

    def evaluate_output_shape(
        self, input_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        return b, self.out_type.size, hi, wi

    def extra_repr(self):
        return "inplace={}, type={}".format(self._inplace, self.in_type)

    def export(self):
        self.eval()

        return torch.nn.LeakyReLU(negative_slope=self._negative_slope, inplace=self._inplace)
