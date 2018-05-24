from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import torch
from gpytorch.module import Module


class Mean(Module):

    # this seems to be identical functionality to what's implemented in the superclass
    # def initialize(self, **kwargs):
    #     for param_name, param_value in kwargs.items():
    #         if hasattr(self, param_name):
    #             if torch.is_tensor(param_value):
    #                 getattr(self, param_name).data.copy_(param_value)
    #             else:
    #                 getattr(self, param_name).data.fill_(param_value)
    #         else:
    #             raise Exception(
    #                 "%s has no parameter %s" % (self.__class__.__name__, param_name)
    #             )
    #     return self

    def forward(self, x):
        raise NotImplementedError()
