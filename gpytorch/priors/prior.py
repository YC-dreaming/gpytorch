from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod, abstractproperty, ABC


class Prior(ABC):

    @abstractmethod
    def extend(self, n):
        raise NotImplementedError()

    @abstractproperty
    def initial_guess(self):
        raise NotImplementedError()

    @abstractmethod
    def is_in_support(self, parameter):
        raise NotImplementedError()

    @property
    def log_transform(self):
        return self._log_transform

    @abstractmethod
    def _log_prob(self, parameter):
        raise NotImplementedError()

    def log_prob(self, parameter):
        return self._log_prob(parameter.exp() if self.log_transform else parameter)

    @abstractproperty
    def size(self):
        raise NotImplementedError()


class TorchDistributionPrior(Prior):

    def _log_prob(self, parameter):
        return sum(
            d.log_prob(p) for d, p in
            zip(self._distributions, parameter.view(self.size))
        )

    @property
    def size(self):
        return len(self._distributions)
