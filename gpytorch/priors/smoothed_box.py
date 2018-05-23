from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from numbers import Number

import torch
from torch.distributions.normal import Normal
from gpytorch.priors.prior import Prior


class SmoothedBoxPrior(Prior):
    """A smoothed approximation of a uniform prior.

    Has full support on the reals and is differentiable everywhere.

        B = {x: a_i <= x_i <= b_i}
        d(x, B) = min_{x' in B} |x - x'|

        pdf(x) ~ exp(- d(x, B)**2 / sqrt(2 * pi * sigma**2))

    """

    def __init__(self, a, b, sigma=0.01, log_transform=False):
        if isinstance(a, Number) and isinstance(b, Number):
            a = torch.tensor([a], dtype=torch.float)
            b = torch.tensor([b], dtype=torch.float)
        elif not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise ValueError("a and b must be both either scalars or Tensors")
        elif a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if torch.any(b < a):
            raise ValueError("must have that a < b (element-wise)")
        self._a = a.type(torch.float)
        self._b = b.type(torch.float)
        if isinstance(sigma, Number):
            self._sigma = torch.full_like(self._a, sigma)
        else:
            self._sigma = sigma.view(self._a.shape)
        self._c = (self._a + self._b) / 2
        self._r = (self._b - self._a) / 2
        self._tails = [Normal(loc=0, scale=s, validate_args=True) for s in self._sigma]
        # normalization factor to make this a probability distribution
        self._M = torch.log(
            1 + (self._b - self._a) / (math.sqrt(2 * math.pi) * self._sigma)
        )
        self._log_transform = log_transform

    def extend(self, n):
        if self.size == n:
            return self
        elif self.size == 1:
            self.__init__(
                a=self._a.repeat(n), b=self._b.repeat(n), sigma=self._sigma.item()
            )
            return self
        else:
            raise ValueError("Can only extend priors of size 1.")

    def _log_prob(self, parameter):
        # x = "distances from box`"
        X = ((parameter.view(self._a.shape) - self._c).abs_() - self._r).clamp(min=0)
        return sum(p.log_prob(x) for x, p in zip(X, self._tails)) - self._M.sum()

    @property
    def initial_guess(self):
        return self._c

    def is_in_support(self, parameter):
        return True

    @property
    def size(self):
        return len(self._a)
