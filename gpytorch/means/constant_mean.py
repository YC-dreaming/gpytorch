from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from .mean import Mean


class ConstantMean(Mean):

    def __init__(self, batch_size=None, prior=None):
        super(ConstantMean, self).__init__()
        self.batch_size = batch_size
        if batch_size is None:
            self.register_parameter(
                "constant", nn.Parameter(torch.zeros(1)), prior=prior
            )
        else:
            self.register_parameter(
                "constant", nn.Parameter(torch.zeros(batch_size, 1), prior=prior)
            )

    def forward(self, input):
        if self.batch_size is None:
            return self.constant.expand(input.size(0))
        else:
            return self.constant.expand(input.size(0), input.size(1))
