from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Number

import torch
from .prior import Prior


class SmoothedUniformPrior(Prior):
    """A smoothed approximation of a uniform prior with differentiable pdf."""

    def __init__(self, a, b, gamma=1e-3, log_transform=False):
        if isinstance(a, Number) and isinstance(b, Number):
            a = torch.tensor([a], dtype=torch.float)
            b = torch.tensor([b], dtype=torch.float)
        elif not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise ValueError("a and b must be both either scalars or Tensors")
        elif a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if torch.any(b < a):
            raise ValueError("must have that a < b (element-wise)")
        if gamma >= 1:
            raise ValueError("gamma must be less than 1")
        self._a = a.type(torch.float)
        self._b = b.type(torch.float)
        self._c = (self._a + self._b) / 2
        self._r = (self._b - self._a) / 2
        self._gamma = gamma
        self._log_transform = log_transform

    def _probs(self, parameter):
        w = (1 + self._gamma) * self._r - (parameter - self._c).abs()
        w.div_(2 * self._gamma * self._r).clamp_(0, 1)
        return 0.5 * self._r * smooth_step(w)

    def _log_prob(self, parameter):
        return torch.log(self._probs(parameter)).sum()

    @property
    def initial_guess(self):
        return self._c

    def is_in_support(self, parameter):
        return bool(
            ((parameter - self._c).abs() < (1 + self._gamma * self._r)).all().item()
        )

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
    return x ** 3 * (6 * x ** 2 - 15 * x + 10)
