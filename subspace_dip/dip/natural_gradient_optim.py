"""
Provides :class:`NGD`.
"""
from typing import Tuple, Dict, Optional

import torch
import numpy as np

from torch import Tensor
from torch.optim import Optimizer

from .fisher_info import FisherInfo

__all__ = ['NGD', 'ngd']

class _RequiredParameter(object):
    """ Singleton class representing a required parameter for an Optimizer.
        https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
    """
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

def _use_grad_for_differentiable(func):
    """ Singleton class representing a required parameter for an Optimizer.
        https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
    """
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad

class NGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0,
            stats_interval=0, curvature_reduction_scale=0., differentiable=False):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=0, 
                    stats_interval=stats_interval, curvature_reduction_scale=curvature_reduction_scale, 
                    differentiable=differentiable
                )
        self.step_cnt = 0
        self.old_step = None 

        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self,
            closure,
            curvature: FisherInfo,
            curvature_kwargs: Dict,
            hyperparams_kwargs: Dict, 
            use_adaptive_learning_rate: bool = False,
            use_adaptive_momentum: bool = False,
            use_adaptive_damping: bool = False,
            use_approximate_quad_model: bool = False,
            return_stats: bool = False
            ):
        
        """Performs a single optimization step.
        Args:
            curvature_matrix
        """

        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params_with_grad = group['params'][0]

        step, loss, output, curvature_reduction_scale, stats = ngd(
            closure=closure,
            params_with_grad=params_with_grad,
            old_step=self.old_step,
            curvature=curvature,
            curvature_kwargs=curvature_kwargs,
            hyperparams_kwargs=hyperparams_kwargs,
            use_adaptive_learning_rate=use_adaptive_learning_rate,
            lr=group['lr'],
            use_adaptive_momentum=use_adaptive_momentum,
            momentum=group['momentum'],
            use_adaptive_damping=use_adaptive_damping,
            use_approximate_quad_model=use_approximate_quad_model,
            weight_decay=group['weight_decay'],
            curvature_reduction_scale=group['curvature_reduction_scale'],
            step_cnt=self.step_cnt,
            stats_interval=group['stats_interval'],
            return_stats=return_stats
            )
        
        self.step_cnt += 1
        self.old_step = step
        self.param_groups[0]['curvature_reduction_scale'] = curvature_reduction_scale

        return loss, output, stats

def ngd(
        closure, 
        params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        hyperparams_kwargs: Dict, 
        old_step: Tensor,
        use_adaptive_learning_rate: bool = False,
        lr: float = 1,
        use_adaptive_momentum: bool = False,
        momentum: float = 0.,
        use_adaptive_damping: bool = False,
        weight_decay: float = 0.,
        curvature_reduction_scale: float = 1., 
        use_approximate_quad_model: bool = False,
        step_cnt: int = 0, 
        stats_interval: int = 20,
        return_stats: bool = False
        ):

    func = _single_tensor_ngd

    outs = func(
        closure=closure,
        params_with_grad=params_with_grad,
        curvature=curvature,
        curvature_kwargs=curvature_kwargs,
        hyperparams_kwargs=hyperparams_kwargs,
        old_step=old_step,
        use_adaptive_learning_rate=use_adaptive_learning_rate,
        lr=lr,
        use_adaptive_momentum=use_adaptive_momentum,
        momentum=momentum,
        use_adaptive_damping=use_adaptive_damping,
        weight_decay=weight_decay,
        curvature_reduction_scale=curvature_reduction_scale,
        use_approximate_quad_model=use_approximate_quad_model,
        step_cnt=step_cnt,
        stats_interval=stats_interval,
        return_stats=return_stats
    )

    return outs

def _single_tensor_ngd(
        closure,
        params_with_grad: Tensor,
        curvature: FisherInfo,
        curvature_kwargs: Dict,
        hyperparams_kwargs: Dict, 
        old_step: Tensor,
        use_adaptive_learning_rate: bool = False,
        lr: float = 1,
        use_adaptive_momentum: bool = False,
        momentum: float = 0.,
        weight_decay: float = 0.,
        curvature_reduction_scale: float = 1., 
        use_adaptive_damping: bool = False,
        use_approximate_quad_model: bool = False, 
        step_cnt: int = 0,
        stats_interval: int = 20,
        return_stats: bool = False
        ):

    # update curvature estimate
    curvature_matrix_update_kwargs =  {
        'num_random_vecs': curvature_kwargs['num_random_vecs'],
        'forward_op_as_part_of_model': curvature_kwargs['forward_op_as_part_of_model'],
        'mode': curvature_kwargs['mode']
    }

    if curvature_kwargs['update_curvature_ema']:
        curvature.update_curvature_ema(
            step_cnt=step_cnt,
            update_kwargs=curvature_kwargs['curvature_ema_kwargs']
            )
        num_random_vecs = np.clip(
            int( curvature_kwargs['num_random_vecs'] * (1 - curvature.curvature_ema) ), 
            a_min=1, a_max=curvature_kwargs['num_random_vecs']
            )
        curvature_matrix_update_kwargs.update({'num_random_vecs': 2 * num_random_vecs})

    curvature.update(**curvature_matrix_update_kwargs)
    # compute loss and proposed directions (i.e. gradients: ∇h(θ = γ(c))) 
    with torch.enable_grad():
        loss, output = closure(parameters_vec=params_with_grad)
        loss.backward()
    
    descent_directions = params_with_grad.grad.detach() # ∇h
    # compute proposed directions (i.e. natrual gradients) -∆ = \tilde{F}^-1∇h
    # a.k.a. preconditioned gradients 
    natural_descent_directions = curvature.approx_fisher_vp(
        descent_directions,
        include_damping=True,
        include_Tikhonov_regularization=True,
        weight_decay=weight_decay,
        use_inverse=True # \tilde{F}^-1
    )

    # vectors = (natural_descent_directions, step) -> preconditioned_gradients, velocities
    # compute the optimal coefficients (αt learning rate and μt momentum coefficients)
    if old_step is None: 
        old_step = torch.zeros_like(natural_descent_directions) + 1e-6
    
    if use_adaptive_learning_rate:

        lr, momentum = _compute_the_optimal_coefficients_via_quad_model(
            curvature=curvature,
            descent_directions=descent_directions,
            natural_descent_directions=natural_descent_directions,
            use_adaptive_momentum=use_adaptive_momentum,
            old_step=old_step,
            weight_decay=weight_decay,
            forward_op_as_part_of_model=curvature_kwargs['forward_op_as_part_of_model'],
            curvature_reduction_scale=curvature_reduction_scale, 
            use_approximate_quad_model=use_approximate_quad_model
        )

        lr = np.clip(lr, 
            a_min=hyperparams_kwargs['min_lr_value'] if hyperparams_kwargs['min_lr_value'] is not None else -np.inf, 
            a_max=hyperparams_kwargs['max_lr_value'] if hyperparams_kwargs['max_lr_value'] is not None else np.inf
            )
        momentum = np.clip(momentum, 
            a_min=hyperparams_kwargs['momentum_min_value'] if hyperparams_kwargs['momentum_min_value'] is not None else -np.inf,
            a_max=1.
            )
    
    # compute delta and return velocities, old_step, a.k.a 𝛿 = -lr*F^-1 ∇h + μ*v
    step = -lr*natural_descent_directions + momentum * old_step
    params_with_grad.add_(step, alpha=1) # update parameters c + 𝛿

    # Optionally compute the reduction ratio and update the damping
    pred_change, change_in_objective = None, None
    if use_adaptive_damping and ((step_cnt + 1) % hyperparams_kwargs['adaptation_interval'] == 0):

        damping_adaptive_kwargs = {
            'weight_decay': weight_decay,
            'adaptation_interval': hyperparams_kwargs['adaptation_interval'],
            'adaptation_decay': curvature_kwargs['adaptive_damping_kwargs']['adaptation_decay'],
            'lower_threshold': curvature_kwargs['adaptive_damping_kwargs']['lower_threshold'],
            'upper_threshold': curvature_kwargs['adaptive_damping_kwargs']['upper_threshold'],
            'min_hyperparam': curvature_kwargs['adaptive_damping_kwargs']['min_value'],
            'max_hyperparam': curvature_kwargs['adaptive_damping_kwargs']['max_value'],
            'use_approximate_quad_model': use_approximate_quad_model
            }
        current_damping = curvature.curvature_damping.damping
        updated_damping, rho, pred_change, change_in_objective = _update_hyperparam_based_on_reduction_ratio(
            curvature=curvature,
            closure=closure,
            old_loss=loss,
            params_with_grad=params_with_grad,
            step=step,
            current_hyperparam=current_damping,
            curvature_reduction_scale=curvature_reduction_scale,
            forward_op_as_part_of_model=curvature_kwargs['forward_op_as_part_of_model'],
            **damping_adaptive_kwargs
        )
        # note that the damping is an attribute of FisherInfo
        curvature.curvature_damping.damping = updated_damping
    
    if (step_cnt + 1) %  hyperparams_kwargs['adaptation_interval'] == 0 and use_adaptive_learning_rate:
        
        curvature_reduction_adaptive_kwargs = {
            'weight_decay': weight_decay,
            'adaptation_interval': hyperparams_kwargs['adaptation_interval'],
            'adaptation_decay': hyperparams_kwargs['curvature_reduction']['adaptation_decay'],
            'lower_threshold': hyperparams_kwargs['curvature_reduction']['lower_threshold'], # a more sentive threshold used here (w.r.t. damping)
            'upper_threshold': hyperparams_kwargs['curvature_reduction']['upper_threshold'], 
            'min_hyperparam': hyperparams_kwargs['curvature_reduction']['min_value'],
            'max_hyperparam': 1.,
            'use_approximate_quad_model': use_approximate_quad_model
            }

        updated_curvature_reduction_scale, rho, _, _ = _update_hyperparam_based_on_reduction_ratio(
            curvature=curvature,
            closure=closure,
            old_loss=loss,
            params_with_grad=params_with_grad,
            step=step,
            pred_change=pred_change, 
            change_in_objective=change_in_objective,
            current_hyperparam=curvature_reduction_scale,
            curvature_reduction_scale=curvature_reduction_scale,
            forward_op_as_part_of_model=curvature_kwargs['forward_op_as_part_of_model'],
            **curvature_reduction_adaptive_kwargs
        )
        curvature_reduction_scale = updated_curvature_reduction_scale

    stats = None
    if return_stats and (step_cnt + 1) % stats_interval == 0:
        stats = {
            'rho': rho.item() if 'rho' in locals() else 0.,
            'model_change': pred_change if 'pred_change' in locals() else 0.,
            'curvature_damping': curvature.curvature_damping.damping,
            'lr': lr,
            'momentum': momentum if hasattr(momentum, 'item') else momentum,
            'curvature_reduction_scale': curvature_reduction_scale,
            'curvature_ema': curvature.curvature_ema,
            'step': step.pow(2).sum().item(), 
            'descent_directions': descent_directions.pow(2).sum().item(),
            'natural_descent_directions_norm': natural_descent_directions.pow(2).sum().item()
        }

    outs = (step, loss.detach(), output.detach(), curvature_reduction_scale, None) if not return_stats \
                else (step, loss.detach(), output.detach(), curvature_reduction_scale, stats)
    
    return outs

def _compute_the_optimal_coefficients_via_quad_model(
        curvature: FisherInfo, 
        natural_descent_directions: Tensor, 
        descent_directions: Tensor,
        old_step: Tensor,
        use_adaptive_momentum: bool = True,
        forward_op_as_part_of_model: bool = True, 
        weight_decay: float = 0., 
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 1., 
    ):

    assert old_step.ndim == 1
    assert old_step.ndim == natural_descent_directions.ndim
    assert old_step.ndim == descent_directions.ndim

    regulariser = curvature.curvature_damping.damping + weight_decay
    Δ = - natural_descent_directions
    if not use_approximate_quad_model:
        JcΔ = curvature.exact_fisher_vp(Δ, use_square_root=True, 
            forward_op_as_part_of_model=forward_op_as_part_of_model).flatten()  
    else: 
        JcΔ = curvature.approx_fisher_vp(Δ, use_square_root=True)

    ΔTΔ = torch.dot(Δ, Δ)
    ΔTFΔ = torch.dot(JcΔ, JcΔ) + (ΔTΔ * regulariser)
    if use_adaptive_momentum:
        if not use_approximate_quad_model:
            Jcold_step = curvature.exact_fisher_vp(
                old_step, use_square_root=True, 
                forward_op_as_part_of_model=forward_op_as_part_of_model
                ).flatten()
        else: 
            Jcold_step = curvature.approx_fisher_vp(old_step, use_square_root=True)
        old_stepTold_step = torch.dot(old_step, old_step)
        old_stepTFold_step = torch.dot(Jcold_step, Jcold_step) + old_stepTold_step*regulariser
        ΔTFold_step = torch.dot(JcΔ, Jcold_step) + torch.dot(Δ, old_step) * regulariser
        sys_matrix = curvature_reduction_scale*torch.Tensor([[ΔTFΔ, ΔTFold_step],[ΔTFold_step, old_stepTFold_step]])
        b = torch.Tensor([torch.dot(descent_directions, Δ), torch.dot(descent_directions, old_step)])
        optimal_coeffs = torch.linalg.solve(-sys_matrix, b)
    else:
        optimal_coeffs = torch.Tensor([- torch.dot(descent_directions, Δ) / (curvature_reduction_scale*ΔTFΔ), 0.])
    assert optimal_coeffs.shape == (2,)

    return optimal_coeffs[0].item(),  optimal_coeffs[1].item()

def _get_quad_model(
        curvature: FisherInfo,
        descent_directions: Tensor, # F^-1 ∇h 
        step: Tensor,
        weight_decay: float,
        forward_op_as_part_of_model: bool = True, 
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 1.
    ) -> Tuple[Tensor, Tensor]:
    
    assert step.ndim == 1
    assert descent_directions.ndim == step.ndim
    regulariser = curvature.curvature_damping.damping + weight_decay
    Jc𝛿 = curvature.exact_fisher_vp(step,
            use_square_root=True, forward_op_as_part_of_model=forward_op_as_part_of_model
                ).flatten() if not use_approximate_quad_model else curvature.approx_fisher_vp(step,
                    use_square_root=True
                    )
    assert Jc𝛿.ndim == 1
    𝛿TF𝛿 = torch.dot(Jc𝛿, Jc𝛿) + torch.dot(step, step) * regulariser
    tangent_plane = torch.dot(descent_directions, step)

    return curvature_reduction_scale*𝛿TF𝛿, tangent_plane

def _compute_quadratic_model_value(
        curvature: FisherInfo,
        descent_directions: Tensor,
        step: Tensor,
        weight_decay: float,
        forward_op_as_part_of_model: bool = True,
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 1.
    ) -> Tensor:

    scaled_𝛿TF𝛿, tangent_plane = _get_quad_model(
        curvature=curvature, 
        descent_directions=descent_directions, 
        step=step, 
        weight_decay=weight_decay,
        forward_op_as_part_of_model=forward_op_as_part_of_model,
        use_approximate_quad_model=use_approximate_quad_model, 
        curvature_reduction_scale=curvature_reduction_scale
        )
    quad_model_change = scaled_𝛿TF𝛿/2 + tangent_plane
    return quad_model_change.item(), scaled_𝛿TF𝛿, tangent_plane

def _update_hyperparam_based_on_reduction_ratio(
        curvature: FisherInfo,
        closure: callable,
        old_loss: Tensor,
        params_with_grad: Tensor,
        step: Tensor,
        current_hyperparam: float,
        pred_change: Optional[Tensor] = None, 
        change_in_objective: Optional[Tensor] = None,
        forward_op_as_part_of_model: bool = True, 
        weight_decay: float = 0.,
        min_hyperparam: float = 1e-0,
        max_hyperparam: float = 100.,
        adaptation_interval: int = 5, 
        adaptation_decay: float = 0.75,
        lower_threshold: float = 0.25,
        upper_threshold: float = 0.75,
        use_approximate_quad_model: bool = False, 
        curvature_reduction_scale: float = 0.001, 
    ) -> Tuple[float,float]:
    

    # reduction ratio
    if pred_change is None and change_in_objective is None: 
        pred_change, _, _ = _compute_quadratic_model_value(
            curvature=curvature,
            # ∇h at this point params_with_grad have been updated but not the grads
            step=step, descent_directions=params_with_grad.grad,
            weight_decay=weight_decay,
            forward_op_as_part_of_model=forward_op_as_part_of_model,
            use_approximate_quad_model=use_approximate_quad_model,
            curvature_reduction_scale=curvature_reduction_scale)

        # at this point params_with_grad have been updated
        new_loss = closure(parameters_vec=params_with_grad)[0]
        change_in_objective = new_loss - old_loss

    rho = change_in_objective / pred_change
    rho_not_nan = torch.nan_to_num(rho, nan=-100.0)

    if rho_not_nan < lower_threshold:
        updated_hyperparam = adaptation_decay**(-adaptation_interval)*current_hyperparam
    elif rho_not_nan > upper_threshold:
        updated_hyperparam = adaptation_decay**(adaptation_interval)*current_hyperparam
    else:
        updated_hyperparam = current_hyperparam
    updated_hyperparam = np.clip(updated_hyperparam, a_min=min_hyperparam, a_max=max_hyperparam)

    return updated_hyperparam, rho_not_nan, pred_change, change_in_objective
