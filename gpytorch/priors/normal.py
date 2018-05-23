from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from torch.distributions.normal import Normal
from gpytorch.priors.prior import TorchDistributionPrior


class NormalPrior(TorchDistributionPrior):

    def __init__(self, loc, scale, log_transform=False):
        if isinstance(loc, Number) and isinstance(scale, Number):
            loc = torch.tensor([loc], dtype=torch.float)
            scale = torch.tensor([scale], dtype=torch.float)
        elif not (torch.is_tensor(loc) and torch.is_tensor(scale)):
            raise ValueError("loc and scale must be both either scalars or Tensors")
        elif loc.shape != scale.shape:
            raise ValueError("loc and scale must have the same shape")
        self._distributions = [
            Normal(loc=lc, scale=sc, validate_args=True)
            for lc, sc in zip(loc, scale)
        ]
        self._log_transform = log_transform

    def extend(self, n):
        if self.size == n:
            return self
        elif self.size == 1:
            loc = self._distributions[0].loc.item()
            scale = self._distributions[0].scale.item()
            self._distributions = [
                Normal(loc=loc, scale=scale, validate_args=True)
                for _ in range(n)
            ]
            return self
        else:
            raise ValueError("Can only extend priors of size 1.")

    @property
    def initial_guess(self):
        return torch.cat([d.mean.view(-1) for d in self._distributions])

    def is_in_support(self, parameter):
        return True
