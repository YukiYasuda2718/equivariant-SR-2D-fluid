from logging import getLogger
from typing import Tuple

from e2cnn import nn as enn
from e2cnn.nn.field_type import FieldType

logger = getLogger()


class EqDSCMS(enn.EquivariantModule):
    """SE2-equivariant DSC/MS"""

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        dsc_feature_type: FieldType,
        ms1_feature_type: FieldType,
        ms2_feature_type: FieldType,
        ms3_feature_type: FieldType,
        ms4_feature_type: FieldType,
    ):
        super(EqDSCMS, self).__init__()

        self.in_type = in_type
        self.out_type = out_type

        # Down-sampled skip-connection model (DSC)
        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            logger.info("dsc1 is pointwise mp")
            self.dsc1_mp = enn.PointwiseMaxPool(in_type, kernel_size=8, padding=0)
        else:
            logger.info("dsc1 is norm mp")
            self.dsc1_mp = enn.NormMaxPool(in_type, kernel_size=8, padding=0)

        self.dsc1_layers = enn.SequentialModule(
            enn.R2Conv(self.dsc1_mp.out_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Upsampling(dsc_feature_type, scale_factor=2),
        )

        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            logger.info("dsc2 is pointwise mp")
            self.dsc2_mp = enn.PointwiseMaxPool(in_type, kernel_size=4, padding=0)
        else:
            logger.info("dsc2 is norm mp")
            self.dsc2_mp = enn.NormMaxPool(in_type, kernel_size=4, padding=0)

        self.dsc2_layers = enn.SequentialModule(
            enn.R2Conv(
                self.dsc2_mp.out_type + self.dsc1_layers.out_type,
                dsc_feature_type,
                kernel_size=3,
                padding=1,
            ),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Upsampling(dsc_feature_type, scale_factor=2),
        )

        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            logger.info("dsc3 is pointwise mp")
            self.dsc3_mp = enn.PointwiseMaxPool(in_type, kernel_size=2, padding=0)
        else:
            logger.info("dsc3 is norm mp")
            self.dsc3_mp = enn.NormMaxPool(in_type, kernel_size=2, padding=0)

        self.dsc3_layers = enn.SequentialModule(
            enn.R2Conv(
                self.dsc3_mp.out_type + self.dsc2_layers.out_type,
                dsc_feature_type,
                kernel_size=3,
                padding=1,
            ),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Upsampling(dsc_feature_type, scale_factor=2),
        )

        self.dsc4_layers = enn.SequentialModule(
            enn.R2Conv(
                in_type + self.dsc3_layers.out_type, dsc_feature_type, kernel_size=3, padding=1
            ),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
        )

        # Multi-scale model (MS)
        _ms1_type = ms1_feature_type + ms1_feature_type
        self.ms1_layers = enn.SequentialModule(
            enn.R2Conv(in_type, _ms1_type, kernel_size=5, padding=2),
            enn.ReLU(_ms1_type, inplace=True),
            enn.R2Conv(_ms1_type, ms1_feature_type, kernel_size=5, padding=2),
            enn.ReLU(ms1_feature_type, inplace=True),
            enn.R2Conv(ms1_feature_type, ms1_feature_type, kernel_size=5, padding=2),
            enn.ReLU(ms1_feature_type, inplace=True),
        )

        _ms2_type = ms2_feature_type + ms2_feature_type
        self.ms2_layers = enn.SequentialModule(
            enn.R2Conv(in_type, _ms2_type, kernel_size=9, padding=4),
            enn.ReLU(_ms2_type, inplace=True),
            enn.R2Conv(_ms2_type, ms2_feature_type, kernel_size=9, padding=4),
            enn.ReLU(ms2_feature_type, inplace=True),
            enn.R2Conv(ms2_feature_type, ms2_feature_type, kernel_size=9, padding=4),
            enn.ReLU(ms2_feature_type, inplace=True),
        )

        _ms3_type = ms3_feature_type + ms3_feature_type
        self.ms3_layers = enn.SequentialModule(
            enn.R2Conv(in_type, _ms3_type, kernel_size=13, padding=6),
            enn.ReLU(_ms3_type, inplace=True),
            enn.R2Conv(_ms3_type, ms3_feature_type, kernel_size=13, padding=6),
            enn.ReLU(ms3_feature_type, inplace=True),
            enn.R2Conv(ms3_feature_type, ms3_feature_type, kernel_size=13, padding=6),
            enn.ReLU(ms3_feature_type, inplace=True),
        )

        _ms4_type = (
            in_type + self.ms1_layers.out_type + self.ms2_layers.out_type + self.ms3_layers.out_type
        )
        self.ms4_layers = enn.SequentialModule(
            enn.R2Conv(_ms4_type, ms4_feature_type, kernel_size=7, padding=3),
            enn.ReLU(ms4_feature_type, inplace=True),
            enn.R2Conv(ms4_feature_type, ms4_feature_type, kernel_size=5, padding=2),
            enn.ReLU(ms4_feature_type, inplace=True),
        )

        # After concatenating DSC and MS
        _mix_type = self.dsc4_layers.out_type + self.ms4_layers.out_type
        self.final_enn_layer = enn.R2Conv(_mix_type, out_type, kernel_size=3, padding=1)

    def _dsc(self, x0):
        x1 = self.dsc1_layers(self.dsc1_mp(x0))
        mp2 = self.dsc2_mp(x0)
        x2 = self.dsc2_layers(enn.tensor_directsum([mp2, x1]))
        mp3 = self.dsc3_mp(x0)
        x3 = self.dsc3_layers(enn.tensor_directsum([mp3, x2]))
        return self.dsc4_layers(enn.tensor_directsum([x0, x3]))

    def _ms(self, x0):
        x1 = self.ms1_layers(x0)
        x2 = self.ms2_layers(x0)
        x3 = self.ms3_layers(x0)
        return self.ms4_layers(enn.tensor_directsum([x0, x1, x2, x3]))

    def forward(self, x):
        x0 = enn.GeometricTensor(x, self.in_type)
        x1 = self._dsc(x0)
        x2 = self._ms(x0)
        x3 = self.final_enn_layer(enn.tensor_directsum([x1, x2]))
        return x3.tensor

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (
            input_shape[0],
            self.out_type.size,
            input_shape[2],
            input_shape[3],
        )
