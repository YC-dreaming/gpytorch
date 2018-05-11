from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .kernel import Kernel


class RBFKernel(Kernel):

    def __init__(
        self,
        ard_num_dims=None,
        lengthscale_prior=None,
        eps=1e-5,
        active_dims=None,
    ):
        super(RBFKernel, self).__init__(
            has_lengthscale=True,
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            active_dims=active_dims,
        )
        self.eps = eps

    def forward(self, x1, x2):
        lengthscales = self.log_lengthscale.exp() + self.eps
        diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).div_(lengthscales)
        return diff.pow_(2).sum(-1).mul_(-0.5).exp_()
