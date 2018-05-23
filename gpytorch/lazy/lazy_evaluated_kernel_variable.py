from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable
from .. import beta_features
import torch


class LazyEvaluatedKernelVariable(LazyVariable):
    def __init__(self, kernel, x1, x2, **params):
        super(LazyEvaluatedKernelVariable, self).__init__(kernel, x1, x2, **params)
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self.is_batch = self.x1.ndimension() == 3

    def _matmul(self, rhs):
        raise RuntimeError('A LazyEvaluatedKernelVariable is not intended to be used directly as a tensor!'
                           ' Call evaluate() first.')

    def _t_matmul(self, rhs):
        raise RuntimeError('A LazyEvaluatedKernelVariable is not intended to be used directly as a tensor!'
                           ' Call evaluate() first.')

    def _quad_form_derivative(self, left_vecs, right_vecs):
        raise RuntimeError('A LazyEvaluatedKernelVariable is not intended to be used directly as a tensor!'
                           ' Call evaluate() first.')

    def diag(self):
        """
        TODO: Can be handled by calling the kernel after creating some new batch dimensions and transposing.

        Hopefully, this can be easily used to add a 'variance_only' prediction mode.
        """
        raise NotImplementedError('Work in progress')

    def evaluate(self):
        """
        NB: This is a meta LazyVariable, in the sense that evaluate can return a LazyVariable if the kernel being
        evaluated does so.
        """
        from ..kernels import Kernel
        if not self.is_batch:
            x1 = self.x1.unsqueeze(0)
            x2 = self.x2.unsqueeze(0)
        else:
            x1 = self.x1
            x2 = self.x2

        res = super(Kernel, self.kernel).__call__(x1, x2)

        if not self.is_batch:
            res = res[0]

        if not isinstance(res, LazyVariable):
            res = NonLazyVariable(res)

        return res

    def __getitem__(self, index):
        index = list(index) if isinstance(index, tuple) else [index]
        ndimension = self.ndimension()
        index += [slice(None, None, None)] * (ndimension - len(index))
        if self.is_batch:
            batch_index = index[0]
            left_index = index[1]
            right_index = index[2]
            return LazyEvaluatedKernelVariable(self.kernel,
                                               self.x1[batch_index, left_index, :],
                                               self.x2[batch_index, right_index, :])
        else:
            left_index = index[0]
            right_index = index[1]
            return LazyEvaluatedKernelVariable(self.kernel,
                                               self.x1[left_index, :],
                                               self.x2[right_index, :])

    def _size(self):
        if self.is_batch:
            return torch.Size((self.x1.size(0), self.x1.size(-2), self.x2.size(-2)))
        else:
            return torch.Size((self.x1.size(-2), self.x2.size(-2)))

    def exact_predictive_mean(
        self, full_mean, train_labels, noise, precomputed_cache=None
    ):
        n_train = train_labels.size(0)
        if precomputed_cache is None:
            train_mean = full_mean[:n_train]
            train_train_covar = self[:n_train, :n_train].evaluate().add_diag(noise)
            precomputed_cache = train_train_covar.inv_matmul(train_labels - train_mean)

        test_mean = full_mean[n_train:]
        test_train_covar = self[n_train:, :n_train].evaluate()
        res = test_train_covar.matmul(precomputed_cache) + test_mean
        return res, precomputed_cache

    def exact_predictive_covar(self, n_train, noise, precomputed_cache=None):
        if self.ndimension() == 3:
            train_train_covar = self[:, :n_train, :n_train].evaluate().add_diag(noise)
            test_train_covar = self[:, n_train:, :n_train].evaluate()
            test_test_covar = self[:, n_train:, n_train:].evaluate()
        else:
            train_train_covar = self[:n_train, :n_train].evaluate().add_diag(noise)
            test_train_covar = self[n_train:, :n_train].evaluate()
            test_test_covar = self[n_train:, n_train:].evaluate()

        if not beta_features.fast_pred_var.on():
            from .matmul_lazy_variable import MatmulLazyVariable

            test_train_covar = test_train_covar.evaluate()
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar).mul(
                -1
            )
            res = test_test_covar + MatmulLazyVariable(
                test_train_covar, covar_correction_rhs
            )
            return res, None

        if precomputed_cache is None:
            train_train_covar_inv_root = train_train_covar.root_inv_decomposition()
            precomputed_cache = self._exact_predictive_covar_inv_quad_form_cache(
                train_train_covar_inv_root, test_train_covar
            )

        from .root_lazy_variable import RootLazyVariable

        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(
            precomputed_cache, test_train_covar
        )
        res = test_test_covar + RootLazyVariable(covar_inv_quad_form_root).mul(-1)
        return res, precomputed_cache
