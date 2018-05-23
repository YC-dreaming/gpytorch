from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from torch.distributions.gamma import Gamma
from gpytorch.priors.prior import TorchDistributionPrior


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
        self._distributions = [
            Gamma(concentration=c, rate=r, validate_args=True)
            for c, r in zip(concentration, rate)
        ]
        self._log_transform = log_transform

    def extend(self, n):
        if self.size == n:
            return self
        elif self.size == 1:
            c = self._distributions[0].concentration.item()
            r = self._distributions[0].rate.item()
            self._distributions = [
                Gamma(concentration=c, rate=r, validate_args=True)
                for _ in range(n)
            ]
            return self
        else:
            raise ValueError("Can only extend priors of size 1.")

    @property
    def initial_guess(self):
        # return mode if it exists, o/w mean
        c = torch.cat([d.concentration.view(-1) for d in self._distributions])
        r = torch.cat([d.rate.view(-1) for d in self._distributions])
        has_mode = (c > 1).type_as(c)
        return (c - has_mode) / r

    def is_in_support(self, parameter):
        return bool((parameter > 0).all().item())
