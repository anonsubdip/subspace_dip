"""
Provides :class:`FisherInfo`.
"""
from typing import Tuple, Optional, Dict
from functools import partial

import torch
import numpy as np

from torch import Tensor
from functorch import vmap, jacrev, vjp, jvp
from torch.utils.data import DataLoader

from .fisher_info_utils import Damping, SamplingProbes

def _ema_polynomial_scheduler(
    step_cnt: int,
    base_curvature_ema: float = 0.5,
    max_iterations: int = 1000, 
    power: float = .99, 
    max_ema: float = 0.95, 
    increase: bool = True
    ) -> float:

    fct = np.clip(step_cnt / max_iterations, a_max=1, a_min=0)
    return np.clip(base_curvature_ema * (
        (1.1 - fct) ** (- power*increase + power*(not increase) )
            ), a_min=1-max_ema, a_max=max_ema)

class FisherInfo:

    def __init__(self, 
        subspace_dip,
        init_damping: float = 1e-3,
        init_curvature_ema: float = 0.95,
        sampling_probes_mode: str = 'row_norm'
        ):

        self.subspace_dip = subspace_dip
        self.matrix = torch.eye(self.subspace_dip.subspace.num_subspace_params, 
            device=self.subspace_dip.device
        )
        self.init_matrix = None
        self.curvature_damping = Damping(init_damping=init_damping)
        self._curvature_ema = init_curvature_ema

        if hasattr(self.subspace_dip.ray_trafo, 'matrix'): 
            prxy_mat = self.subspace_dip.ray_trafo.matrix 
        else:
            prxy_mat = None 
            assert sampling_probes_mode != 'row_norm'

        self.probes = SamplingProbes(
                prxy_mat=prxy_mat,
                mode=sampling_probes_mode, 
                device=self.subspace_dip.device
            )
    
    @property
    def shape(self, ) -> Tuple[int,int]:
        size = self.matrix.shape[0]
        return (size, size)
    
    @property
    def curvature_ema(self, ) -> float:
        return self._curvature_ema 

    @curvature_ema.setter
    def curvature_ema(self, value) -> None:
        self._curvature_ema = value
    
    def update_curvature_ema(self, 
        step_cnt: int, 
        update_kwargs: Dict 
        ) -> None:
        self.curvature_ema = _ema_polynomial_scheduler(step_cnt=step_cnt, **update_kwargs)

    def reset_fisher_matrix(self, ) -> None:
        self.matrix = self.init_matrix.clone() if self.init_matrix is not None else torch.eye(
            self.subspace_dip.subspace.num_subspace_params, device=self.subspace_dip.device)

    def approx_fisher_vp(self,
        v: Tensor,
        use_inverse: bool = False,
        use_square_root: bool = False,
        include_damping: bool = False, # include λ
        include_Tikhonov_regularization: bool = False, # include η
        weight_decay: float = 0.
        ) -> Tensor:

        matrix = self.matrix.clone() if (
            include_damping or include_Tikhonov_regularization
                ) else self.matrix
        if use_square_root:
            # it returns the upper triangular matrix 
            chol = torch.linalg.cholesky(matrix, upper=True) 
            return chol@v 
        else:
            if include_damping:
                matrix = self.curvature_damping.add_damping(matrix=matrix, 
                    include_Tikhonov_regularization=include_Tikhonov_regularization, 
                    weight_decay=weight_decay
                ) # add λ and η
            return matrix @ v if not use_inverse else torch.linalg.solve(matrix, v)

    def exact_fisher_vp(self,             
        v: Tensor,
        forward_op_as_part_of_model: bool = True,
        use_square_root: bool = False
        ) -> Tensor:

            _fnet_single = partial(self._fnet_single, forward_op_as_part_of_model=forward_op_as_part_of_model)
            _, jvp_ = jvp(_fnet_single, 
                    (self.subspace_dip.subspace.parameters_vec,), (v,)
                ) # jvp_ = v @ J_cT = v @ (UT JT AT)
            if not forward_op_as_part_of_model:
                jvp_ = self.subspace_dip.ray_trafo(jvp_)
            if not use_square_root: 
                _, _vjp_fn = vjp(_fnet_single,
                            self.subspace_dip.subspace.parameters_vec
                        )
                if not forward_op_as_part_of_model: 
                    jvp_ = self.subspace_dip.ray_trafo.trafo_adjoint(jvp_)
                Fv = _vjp_fn(jvp_)[0] # Fv = jvp_ @ J_c = jvp_ @ (A J U)
            else: 
                Fv = jvp_
            return Fv

    def _fnet_single(self,
        parameters_vec: Tensor,
        input: Optional[Tensor] = None,
        forward_op_as_part_of_model: bool = True
        ) -> Tensor:

        out = self.subspace_dip.forward(
                parameters_vec=parameters_vec,
                input=input,
                **{'apply_forward_op':forward_op_as_part_of_model}
            )
        return out

    def exact_fisher_assembly(self, 
        dataset: Optional[DataLoader] = None,
        forward_op_as_part_of_model: bool = True
        ) -> Tensor:

        def _per_input_exact_update(fbp: Optional[Tensor] = None): 
                
            _fnet_single = partial(self._fnet_single, forward_op_as_part_of_model=forward_op_as_part_of_model)
            if fbp is None:
                jac = jacrev(_fnet_single)(self.subspace_dip.subspace.parameters_vec)
            else:
                jac = vmap(jacrev(_fnet_single), in_dims=(None, 0))(
                    self.subspace_dip.subspace.parameters_vec, fbp.unsqueeze(dim=1)
                )
    
            jac = jac.view(fbp.shape[0] if fbp is not None else 1, #batch_size,
                    -1, self.subspace_dip.subspace.num_subspace_params
                ) # the inferred dim is im_shape: nn_model_output
            
            if not forward_op_as_part_of_model:
                perm_jac = torch.permute(jac, (1, 0, 2)).view(jac.shape[1], -1)
                perm_jac = self.subspace_dip.ray_trafo.trafo_flat(perm_jac)
                jac = perm_jac.view(perm_jac.shape[0], jac.shape[0], jac.shape[2])
                jac = torch.permute(jac, (1, 0, 2))

            return jac # (batch_size, nn_model_output, num_subspace_params)
        
        with torch.no_grad():

            per_inputs_jac_list = []
            if dataset is not None:
                for _, _, fbp in dataset:
                    jac = _per_input_exact_update(fbp=fbp)
                    per_inputs_jac_list.append(jac)
            else:
                jac = _per_input_exact_update(fbp=None)
                per_inputs_jac_list.append(jac)

            per_inputs_jac = torch.cat(per_inputs_jac_list)
            # same as (per_inputs_jac.mT @ per_inputs_jac).mean(dim=0)
            matrix = torch.mean(torch.einsum('Nop,Noc->Npc', per_inputs_jac, per_inputs_jac), dim=0)

            return matrix, jac

    def initialise_fisher_info(self,
        dataset: Optional[DataLoader] = None,
        num_random_vecs: int = 100, 
        forward_op_as_part_of_model: bool = True,
        mode: str = 'full'
        ) -> Tensor:
        
        if mode == 'full':
            matrix, _ = self.exact_fisher_assembly(dataset=dataset,
                forward_op_as_part_of_model=forward_op_as_part_of_model
            )
        elif mode == 'vjp_rank_one':
            matrix = self.online_fisher_assembly(dataset=dataset,
                num_random_vecs=num_random_vecs, forward_op_as_part_of_model=forward_op_as_part_of_model
            )
        else: 
            raise NotImplementedError
        self.matrix = matrix
        self.init_matrix = matrix

    def online_fisher_assembly(self,
        dataset: Optional[DataLoader] = None,
        num_random_vecs: int = 10,
        forward_op_as_part_of_model: bool = True
        ) -> Tensor:

        def _per_input_rank_one_update(fbp: Optional[Tensor] = None):
            
            obs_shape = self.subspace_dip.ray_trafo.obs_shape
            im_shape = self.subspace_dip.ray_trafo.im_shape

            v = self.probes.sample_probes(
                num_random_vecs=num_random_vecs, 
                shape=self.subspace_dip.ray_trafo.obs_shape
                )

            if not forward_op_as_part_of_model: 
                v = self.subspace_dip.ray_trafo.trafo_adjoint(
                        v.view(v.shape[0]*v.shape[1], v.shape[2], *obs_shape)
                    ).view(*v.shape[0:3], *im_shape)

            _fnet_single = partial(self._fnet_single,
                input=fbp,
                forward_op_as_part_of_model=forward_op_as_part_of_model
                )
            _, _vjp_fn = vjp(_fnet_single,
                    self.subspace_dip.subspace.parameters_vec
                )

            def _single_vjp(v):
                return _vjp_fn(v)[0]

            vJp = vmap(_single_vjp, in_dims=0)(v) #vJp = v.T@(AJU)
            matrix = torch.einsum('Np,Nc->pc', vJp, vJp) / num_random_vecs
            return matrix

        with torch.no_grad():

            per_inputs_fisher_list = []
            if dataset is not None: 
                for _, _, fbp in dataset:
                    matrix = _per_input_rank_one_update(fbp=fbp)
                    per_inputs_fisher_list.append(matrix)
            else:
                matrix = _per_input_rank_one_update(fbp=None)
                per_inputs_fisher_list.append(matrix)

            per_inputs_fisher = torch.stack(per_inputs_fisher_list)
            matrix = torch.mean(per_inputs_fisher, dim=0)

        return matrix

    def update(self,
        dataset: Optional[DataLoader] = None,
        num_random_vecs: Optional[int] = 10,
        forward_op_as_part_of_model: bool = True,
        mode: str = 'full'
        ) -> None:

        if mode == 'full':
            update, _ = self.exact_fisher_assembly(
                dataset=dataset, forward_op_as_part_of_model=forward_op_as_part_of_model
            )
        elif mode == 'vjp_rank_one':
            update = self.online_fisher_assembly(dataset=dataset,
                forward_op_as_part_of_model=forward_op_as_part_of_model, num_random_vecs=num_random_vecs
            )
        else:
            raise NotImplementedError
        matrix = self.curvature_ema * self.matrix + (1. - self.curvature_ema) * update
        self.matrix = matrix