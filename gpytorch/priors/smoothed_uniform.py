from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from .prior import Prior


class SmoothedUniformPrior(Prior):

    def __init__(self, a, b, gamma=1e-3):
        if isinstance(a, Number) and isinstance(b, Number):
            a = torch.tensor([a])
            b = torch.tensor([b])
        elif not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise ValueError("a and b must be both either scalars or Tensors")
        elif a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if torch.any(b < a):
            raise ValueError("must have that a < b (element-wise)")
        self._a = a.type(torch.float)
        self._b = b.type(torch.float)
        self._gamma = self._a.new(self._a.shape, gamma)
        self._c = (self._a + self._b) / 2
        self._r = (self._b - self._a) / 2

    def _prob(self, parameter):
        z = (parameter - self._c).abs()
        mask_1 = z < self._r - self._epsilon
        mask_0 = z > self._r + self._epsilon
        mask = ~mask_1 & ~mask_0
        return (1 / 2 * self._r) * (
            mask_1.type(torch.float)
            + mask.type(torch.float)
            * smooth_step((self._r + self._epsilon - z) / 2 / self._epsilon)
        )

    def _log_prob(self, parameter):
        return torch.log(self._prob(parameter)).sum()

    def shape_as(self, tensor):
        if not self.shape == tensor.shape:
            try:
                a_new = self._a.view_as(tensor)
                b_new = self._b.view_as(tensor)
            except RuntimeError:
                raise ValueError("Prior and parameter have incompatible shapes.")
            self.__init__(a=a_new, b=b_new, gamma=self._gamma)
        return self

    @property
    def shape(self):
        return self._a.shape


def smooth_step(x):
    return x**3 * (6 * x**2 - 15 * x + 10)
