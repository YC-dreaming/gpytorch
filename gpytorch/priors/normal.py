from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from torch.distributions.normal import Normal
from .prior import TorchDistributionPrior


class NormalPrior(TorchDistributionPrior):

    def __init__(self, loc, scale, log_transform=False):
        if isinstance(loc, Number) and isinstance(scale, Number):
            loc = torch.tensor([loc])
            scale = torch.tensor([scale])
        elif not (torch.is_tensor(loc) and torch.is_tensor(scale)):
            raise ValueError("loc and scale must be both either scalars or Tensors")
        elif loc.shape != scale.shape:
            raise ValueError("loc and scale must have the same shape")
        self._distribution = Normal(
            loc=loc.type(torch.float), scale=scale.type(torch.float)
        )
        self._log_transform = log_transform

    def shape_as(self, tensor):
        if not self.shape == tensor.shape:
            try:
                loc_new = self.distribution.loc.view_as(tensor)
                scale_new = self.distribution.scale.view_as(tensor)
            except RuntimeError:
                raise ValueError("Prior and parameter have incompatible shapes.")
            self._distribution = Normal(loc=loc_new, scale=scale_new)
        return self

    @property
    def shape(self):
        return self.distribution.loc.shape
