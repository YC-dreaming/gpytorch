from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from numbers import Number

import torch
from torch.distributions.normal import Normal
from .prior import Prior


class SmoothedBoxPrior(Prior):
    """A smoothed approximation of a uniform prior.

    Has full support on the reals and is differentiable everywhere.

        B = {x: a_i <= x_i <= b_i}
        d(x, B) = min_{x' in B} |x - x'|

        pdf(x) ~ exp(- d(x, B)**2 / sqrt(2 * pi * sigma**2))

    """

    def __init__(self, a, b, sigma=1e-2, log_transform=False):
        if isinstance(a, Number) and isinstance(b, Number):
            a = torch.tensor([a], dtype=torch.float)
            b = torch.tensor([b], dtype=torch.float)
        elif not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise ValueError("a and b must be both either scalars or Tensors")
        elif a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if torch.any(b < a):
            raise ValueError("must have that a < b (element-wise)")
        if len(a) > 1:
            # TODO: Implement for multi-dimensional parameters
            raise NotImplementedError(
                "Multi-dimensional smoothed box priors not yet supported."
            )
        self._a = a.type(torch.float)
        self._b = b.type(torch.float)
        self._sigma = self._a.new_tensor(sigma).view(*self._a.shape)
        self._c = (self._a + self._b) / 2
        self._r = (self._b - self._a) / 2
        self._tail = Normal(loc=0, scale=self._sigma, validate_args=True)
        # normalization factor to make this a probability distribution
        self._M = torch.log(
            1 + (self._b - self._a) / (math.sqrt(2 * math.pi) * self._sigma)
        )
        self._log_transform = log_transform

    def _log_prob(self, parameter):
        # x = "distance from box`"
        x = ((parameter - self._c).abs_() - self._r).clamp(min=0)
        return self._tail.log_prob(x) - self._M

    @property
    def initial_guess(self):
        return self._c

    def is_in_support(self, parameter):
        return True

    def shape_as(self, tensor):
        if not self.shape == tensor.shape:
            try:
                a_new = self._a.view_as(tensor)
                b_new = self._b.view_as(tensor)
            except RuntimeError:
                raise ValueError("Prior and parameter have incompatible shapes.")
            self.__init__(a=a_new, b=b_new, sigma=self._sigma)
        return self

    @property
    def shape(self):
        return self._a.shape
