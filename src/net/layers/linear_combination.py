"""

`layers.py`
------------

    This script contains the layers used in the model

"""
import torch
import torch.nn as nn


class LinearCombinationLayer(nn.Module):
    def __init__(
            self,
            filter_order
            ):
        super(LinearCombinationLayer, self).__init__()

        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(filter_order)]
            )

    def forward(self, hs):
        output = 0
        for i, param in enumerate(self.params):
            output = output + param * hs[i]
        return output