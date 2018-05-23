from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from .prior import TorchDistributionPrior


class MultivariateNormalPrior(TorchDistributionPrior):

    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        log_transform=False,
    ):
        if not torch.is_tensor(loc):
            raise ValueError("loc must be a torch Tensor")
        self._distribution = MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=precision_matrix,
            validate_args=True,
        )
        self._log_transform = log_transform

    def _log_prob(self, parameter):
        return self._distribution.log_prob(parameter.view(self.size))

    def extend(self, n):
        if self.size == n:
            return self
        else:
            raise NotImplementedError("Cannot extend MultivariateNormalPrior.")

    @property
    def initial_guess(self):
        return self.distribution.mean

    def is_in_support(self, parameter):
        return True

    @property
    def size(self):
        return len(self._distribution.loc)
