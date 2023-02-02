"""
Provides utils classes for `FisherInfo`.
"""
from typing import Tuple

import numpy as np
import torch

from torch import Tensor

class Damping:

    def __init__(self, init_damping: float = 1e2):
        self._damping = init_damping
    
    @property
    def damping(self, ): 
        return self._damping

    @damping.setter
    def damping(self, value): 
        self._damping = value 

    def add_damping(self,  
        matrix: Tensor,
        include_Tikhonov_regularization: bool = True,
        weight_decay: float = 0.
        ) -> Tensor:

        matrix[np.diag_indices(matrix.shape[0])] += self.damping
        if include_Tikhonov_regularization: 
            matrix[np.diag_indices(matrix.shape[0])] += weight_decay

        return matrix

class SamplingProbes:

    def __init__(self, prxy_mat=None, mode='row_norm', device=None):
        self.prxy_mat = prxy_mat
        self.mode = mode
        self.device = device
        
        if self.mode == 'row_norm':
            assert self.prxy_mat is not None
            assert  self.prxy_mat.ndim == 2

            if not self.prxy_mat.is_sparse:
                un_ps = torch.linalg.norm(self.prxy_mat, dim=1, ord=2).pow(2)
            else:
                un_ps = torch.sparse.sum(self.prxy_mat**2, dim=1).to_dense()            
            const = un_ps.sum()
            self.ps = un_ps/const

        elif self.mode == 'gauss': 
            pass
        else: 
            raise NotImplementedError

    def sample_probes(self, num_random_vecs: int, shape: Tuple[int, int]) -> Tensor:

        if self.mode == 'row_norm': 
            func = self._scaled_unit_probes

        elif self.mode == 'gauss':
            def _gauss_probes(num_random_vecs, shape):
                return torch.randn(
                    (num_random_vecs, 1, 1, *shape), 
                        device=self.device)
            func = _gauss_probes
        else:
            raise NotImplementedError

        return func(num_random_vecs=num_random_vecs, shape=shape)

    def _scaled_unit_probes(self, num_random_vecs, shape) -> Tensor:

        new_shape = (num_random_vecs, 1, 1, np.prod(shape))
        v = torch.zeros(*new_shape, device=self.device)
        rand_inds = np.random.choice(
                np.prod(shape), 
                size=num_random_vecs, 
                replace=True, 
                p=self.ps.cpu().numpy()
            )
        ps_mask = self.ps.expand(num_random_vecs, -1).view(new_shape).pow(-.5)
        v[range(num_random_vecs), :, :, rand_inds] = ps_mask[range(num_random_vecs), :, :, rand_inds]

        return v.reshape(num_random_vecs, 1, 1, *shape)
