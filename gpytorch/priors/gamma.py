from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from torch.distributions.gamma import Gamma
from .prior import TorchDistributionPrior


class GammaPrior(TorchDistributionPrior):

    def __init__(self, concentration, rate, log_transform=False):
        if isinstance(concentration, Number) and isinstance(rate, Number):
            concentration = torch.tensor([concentration])
            rate = torch.tensor([rate])
        elif not (torch.is_tensor(concentration) and torch.is_tensor(rate)):
            raise ValueError(
                "concentration and rate must be both either scalars or Tensors"
            )
        elif concentration.shape != rate.shape:
            raise ValueError("concentration and rate must have the same shape")
        self._distribution = Gamma(
            concentration=concentration.type(torch.float), rate=rate.type(torch.float)
        )
        self._log_transform = log_transform

    def shape_as(self, tensor):
        if not self.shape == tensor.shape:
            try:
                concentration_new = self.distribution.concentration.view_as(tensor)
                rate_new = self.distribution.rate.view_as(tensor)
            except RuntimeError:
                raise ValueError("Prior and parameter have incompatible shapes.")
            self._distribution = Gamma(concentration=concentration_new, rate=rate_new)
        return self

    @property
    def shape(self):
        return self.distribution.concentration.shape
