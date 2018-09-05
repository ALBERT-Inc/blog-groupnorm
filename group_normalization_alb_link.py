import chainer
import numpy as np
from chainer.initializers import _get_initializer

from group_normalization_alb1_func import group_normalization as gn_alb1
from group_normalization_alb2_func import group_normalization as gn_alb2

__all__ = ['GroupNormalizationAlb1', 'GroupNormalizationAlb2']


class _GroupNormalization(chainer.Link):
    def __init__(self, size, groups=32, eps=1e-5, dtype=np.float32,
                 initial_gamma=None, initial_beta=None):
        super().__init__()

        self.groups = groups
        self.eps = eps

        with self.init_scope():
            if initial_gamma is None:
                initial_gamma = 1
            initial_gamma = _get_initializer(initial_gamma)
            initial_gamma.dtype = dtype
            self.gamma = chainer.Parameter(initial_gamma, size)

            if initial_beta is None:
                initial_beta = 0
            initial_beta = _get_initializer(initial_beta)
            initial_beta.dtype = dtype
            self.beta = chainer.Parameter(initial_beta, size)


class GroupNormalizationAlb1(_GroupNormalization):
    def __call__(self, x):
        return gn_alb1(x, self.gamma, self.beta, self.groups, self.eps)


class GroupNormalizationAlb2(_GroupNormalization):
    def __call__(self, x):
        return gn_alb2(x, self.gamma, self.beta, self.groups, self.eps)
