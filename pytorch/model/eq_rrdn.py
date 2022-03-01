from typing import Tuple

from e2cnn import nn as enn
from e2cnn.nn.field_type import FieldType
from eq_leaky_relu import LeakyReLU


class DenseLayer(enn.EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        kernel_size: int,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super(DenseLayer, self).__init__()

        self.conv = enn.R2Conv(in_type, out_type, kernel_size, padding=kernel_size // 2, bias=bias)
        self.nl = LeakyReLU(self.conv.out_type, negative_slope=alpha, inplace=True)

        self.in_type = in_type
        self.out_type = in_type + out_type

    def forward(self, x):
        return enn.tensor_directsum([x, self.nl(self.conv(x))])

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (
            input_shape[0],
            self.out_type.size,
            input_shape[2],
            input_shape[3],
        )


class RDB(enn.EquivariantModule):
    """Equivariant version of Residual Dense Block (RDB)"""

    def __init__(
        self,
        in_type: FieldType,
        growth_type: FieldType,
        num_layers: int,
        kernel_size: int,
        beta: float,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super(RDB, self).__init__()

        self.in_type = in_type
        self.out_type = in_type
        self.beta = beta

        self.layers = []
        _in_type = in_type
        for _ in range(num_layers):
            _d = DenseLayer(_in_type, growth_type, kernel_size, alpha, bias)
            _in_type = _d.out_type
            self.layers.append(_d)
        _in_type = self.layers[-1].out_type
        self.layers = enn.SequentialModule(*self.layers)

        # local feature fusion (lff)
        self.lff = enn.R2Conv(
            _in_type, self.out_type, kernel_size, padding=kernel_size // 2, bias=bias
        )

    def forward(self, x):
        return x + self.beta * self.lff(self.layers(x))  # local residual learning

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (
            input_shape[0],
            self.out_type.size,
            input_shape[2],
            input_shape[3],
        )


class RRDB(enn.EquivariantModule):
    """Equivariant version of Residual in Residual Dense Block (RRDB)"""

    def __init__(
        self,
        in_type: FieldType,
        growth_type: FieldType,
        num_layers: int,
        kernel_size: int,
        num_blocks: int,
        beta: float,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super(RRDB, self).__init__()

        self.in_type = in_type
        self.out_type = in_type
        self.beta = beta

        self.blocks = enn.SequentialModule(
            *[
                RDB(in_type, growth_type, num_layers, kernel_size, beta, alpha, bias)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        return x + self.beta * self.blocks(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (
            input_shape[0],
            self.out_type.size,
            input_shape[2],
            input_shape[3],
        )


class EqRRDN(enn.EquivariantModule):
    """Equivariant version of Residual in Residual Dense Network (RRDN)"""

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        feature_type: FieldType,
        post_feature_type: FieldType,
        growth_type: FieldType,
        num_layers: int = 4,
        kernel_size: int = 3,
        num_blocks: int = 3,
        beta: float = 0.2,
        alpha: float = 0.01,
        bias: bool = True,
    ):
        super(EqRRDN, self).__init__()

        self.in_type = in_type
        self.out_type = out_type
        self.beta = beta
        padding = kernel_size // 2

        # shallow feature extractor (sfe)
        self.sfe_conv = enn.R2Conv(in_type, feature_type, kernel_size, padding=padding, bias=bias)
        self.sfe_nl = LeakyReLU(self.sfe_conv.out_type, negative_slope=alpha, inplace=True)

        self.rrdb = RRDB(
            feature_type, growth_type, num_layers, kernel_size, num_blocks, beta, alpha, bias
        )

        # global feature fusion (gff)
        self.gff = enn.R2Conv(feature_type, feature_type, kernel_size, padding=padding, bias=bias)

        # post process (pp)
        self.pp_layers = enn.SequentialModule(
            enn.R2Conv(feature_type, post_feature_type, kernel_size, padding=padding, bias=bias),
            LeakyReLU(post_feature_type, negative_slope=alpha, inplace=True),
            enn.R2Conv(
                post_feature_type, post_feature_type, kernel_size, padding=padding, bias=bias
            ),
            LeakyReLU(post_feature_type, negative_slope=alpha, inplace=True),
            enn.R2Conv(post_feature_type, feature_type, kernel_size, padding=padding, bias=bias),
            LeakyReLU(feature_type, negative_slope=alpha, inplace=True),
            enn.R2Conv(feature_type, out_type, kernel_size, padding=padding, bias=bias),
        )

    def forward(self, x):
        f = self.sfe_nl(self.sfe_conv(enn.GeometricTensor(x, self.in_type)))
        y = self.rrdb(f)
        y = f + self.beta * self.gff(y)
        y = self.pp_layers(y)
        return y.tensor

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (
            input_shape[0],
            self.out_type.size,
            input_shape[2],
            input_shape[3],
        )
