"""
Provides :class:`SubspaceDeepImagePrior`.
"""
from typing import Optional, Union, Tuple, Dict, Sequence

import os
import socket
import datetime
import torch
import numpy as np
import functorch as ftch
import torch.nn as nn
import tensorboardX

from warnings import warn
from torch import Tensor
from torch.nn import MSELoss
from copy import deepcopy
from tqdm import tqdm
from functools import partial

from subspace_dip.utils import tv_loss, PSNR, SSIM, normalize
from subspace_dip.data import BaseRayTrafo
from .base_dip_image_prior import BaseDeepImagePrior
from .linear_subspace import LinearSubspace
from .fisher_info import FisherInfo
from .natural_gradient_optim import NGD
from .utils import stats_to_writer
from .early_stopping_criteria import EarlyStop

class SubspaceDeepImagePrior(BaseDeepImagePrior, nn.Module):

    def __init__(self,
        subspace: LinearSubspace,
        ray_trafo: BaseRayTrafo,
        state_dict: Optional[None] = None,
        torch_manual_seed: Union[int, None] = 1,
        device=None,
        net_kwargs=None,
        ):

        nn.Module.__init__(self, )
        BaseDeepImagePrior.__init__(self, 
            ray_trafo=ray_trafo,
            torch_manual_seed=torch_manual_seed,
            device=device,
            net_kwargs=net_kwargs
        )
        
        self.subspace = subspace
        if state_dict is not None: 
            self.nn_model.load_state_dict(
                state_dict=state_dict
            )
        self.func_model_with_input, _ = ftch.make_functional(self.nn_model)
        self.pretrained_weights = torch.cat(
            [param.flatten().detach() for param in self.nn_model.parameters()]
        )

    def get_func_params(self, 
            parameters_vec: Optional[Tensor] = None,
            slicing_sequence: Optional[Sequence] = None, 
        ) -> Tuple[Tensor]:

        weights = self.pretrained_weights.clone() # dim_p
        if slicing_sequence is None: # θ = γ(c) = θ_p + \sum_i c_i * u_i; u_i is num_params or len(self.subspace.indices) if self.subspace.is_trimmed
            sub_weights = torch.inner(
                    self.subspace.parameters_vec if parameters_vec is None \
                        else parameters_vec, self.subspace.ortho_basis # parameters_vec: rank_subspace; self.subspace.ortho_basis: num_params * rank_subspace
                    )
        else:
            sub_weights = torch.inner(
                    self.subspace.parameters_vec[slicing_sequence] if parameters_vec is None \
                        else parameters_vec[slicing_sequence], self.subspace.ortho_basis[:, slicing_sequence]
                    )

        if self.subspace.is_trimmed:
            weights[self.subspace.indices] = weights[self.subspace.indices] + sub_weights
        else:
            weights = weights + sub_weights
        cnt = 0
        func_weights = []
        for params in self.nn_model.parameters():
            func_weights.append(
                weights[cnt:cnt+params.numel()].view(params.shape)
            )
            cnt += params.numel()
        return tuple(func_weights)

    def set_nn_model_require_grad(self, set_require_grad: bool = False):
        for params in self.nn_model.parameters():
            params.requires_grad_(set_require_grad)

    def forward(self, 
            parameters_vec: Optional[Tensor] = None, 
            input: Optional[Tensor] = None, 
            slicing_sequence: Optional[Sequence] = None, 
            apply_forward_op: bool = True
        ) -> Tensor:

        out = self.func_model_with_input(
                # θ = γ(c) = θ_p + \sum_i c_i * u_i, parameters_vec = c_i
                self.get_func_params( 
                    parameters_vec=self.subspace.parameters_vec if parameters_vec is None else parameters_vec,
                    slicing_sequence=slicing_sequence
                        ), 
                self.net_input if input is None else input
            )
        return out if not apply_forward_op else self.ray_trafo(out)

    def objective(self,
        criterion,
        noisy_observation: Tensor,
        use_tv_loss: Optional[bool] = None,
        parameters_vec: Optional[Tensor] = None, 
        slicing_sequence: Optional[Sequence] = None,
        apply_forward_op: bool = False,
        weight_decay: Optional[float] = None, 
        gamma: Optional[float] = None, 
        return_output: Optional[bool] = True
        ):

        output = self.forward(
                parameters_vec=parameters_vec,
                slicing_sequence=slicing_sequence,
                apply_forward_op=apply_forward_op
            )

        loss = criterion(self.ray_trafo(output), noisy_observation)
        if use_tv_loss:
            loss = loss + gamma*tv_loss(output)
        if weight_decay !=0:
            loss = loss +  weight_decay / 2 * torch.sum(
                self.subspace.parameters_vec**2 if parameters_vec is None else parameters_vec**2)
        return loss if not return_output else (loss, output)

    def reconstruct(self,
        noisy_observation: Tensor,
        filtbackproj: Optional[Tensor] = None,
        ground_truth: Optional[Tensor] = None,
        recon_from_randn: bool = False,
        use_tv_loss: bool = True,
        fisher_info: Optional[FisherInfo] = None,
        log_path: str = '.',
        show_pbar: bool = True,
        optim_kwargs: Dict = None
        ) -> Tensor:

        writer = tensorboardX.SummaryWriter(
                logdir=os.path.join(log_path, '_'.join((
                        datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        socket.gethostname(),
                        '_Subspace_DIP' if not use_tv_loss else '_Subspace_DIP+TV'))))

        self.set_nn_model_require_grad(False)
        self.nn_model.train()

        self.net_input = (
            0.1 * torch.randn(1, 1, *self.ray_trafo.im_shape, device=self.device)
            if recon_from_randn else
            filtbackproj.to(self.device)
        )

        if optim_kwargs['optim']['optimizer'] == 'adam':

            self.optimizer = torch.optim.Adam(
                [self.subspace.parameters_vec],
                lr=optim_kwargs['optim']['lr'],
                weight_decay=optim_kwargs['optim']['weight_decay']
                )

        elif optim_kwargs['optim']['optimizer'] == 'lbfgs':

            self.optimizer = torch.optim.LBFGS(
                [self.subspace.parameters_vec],
                line_search_fn="strong_wolfe"
                )

        elif optim_kwargs['optim']['optimizer'] == 'ngd':

            self.optimizer = NGD(
                [self.subspace.parameters_vec],
                lr=optim_kwargs['optim']['lr'],
                weight_decay=optim_kwargs['optim']['weight_decay'], 
                momentum=optim_kwargs['optim']['momentum'],
                stats_interval=optim_kwargs['optim']['stats_interval'], 
                curvature_reduction_scale=optim_kwargs['optim']['init_scale_curvature']
                )
            curvature_update_kwargs = {
                'num_random_vecs': optim_kwargs['optim']['num_random_vecs'],
                'forward_op_as_part_of_model': optim_kwargs['optim']['forward_op_as_part_of_model'],
                'mode': optim_kwargs['optim']['mode'], 
                'update_curvature_ema': optim_kwargs['optim']['update_curvature_ema'],
                'adaptive_damping_kwargs': optim_kwargs['optim']['adaptive_damping_kwargs'],
            }
            if optim_kwargs['optim']['update_curvature_ema']:
                curvature_update_kwargs.update(
                            {'curvature_ema_kwargs': optim_kwargs['optim']['curvature_ema_kwargs']})
            ngd_hyperparams_kwargs = optim_kwargs['optim']['hyperparams_kwargs']

        else:
            raise NotImplementedError

        noisy_observation = noisy_observation.to(self.device)
        if optim_kwargs['loss_function'] == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        min_loss_state = {
            'loss': np.inf,
            'output': self.nn_model(self.net_input).detach(),  # pylint: disable=not-callable
            'params_state_dict': deepcopy(self.subspace.state_dict()),
        }

        if ground_truth is not None:
            writer.add_image('ground_truth', normalize(
               ground_truth[0, ...]).cpu().numpy(), 0)

        if filtbackproj is not None: 
            writer.add_image('filtbackproj', normalize(
                filtbackproj[0, ...]).cpu().numpy(), 0)

        if optim_kwargs['optim']['use_subsampling_orthospace']:
            self.singular_probabilities = self.subspace.singular_values / torch.sum(self.subspace.singular_values)

        writer.add_image('base_recon', normalize(
               self.nn_model(self.net_input)[0, ...].detach().cpu().numpy()), 0)
        
        print('Pre-trained UNET reconstruction of sample')
        print('PSNR:', PSNR(self.nn_model(self.net_input)[0, 0].detach().cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(self.nn_model(self.net_input)[0, 0].detach().cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        if optim_kwargs['early_stop']['use_early_stop']: 
            earlystop = EarlyStop(size=optim_kwargs['early_stop']['buffer_size'], patience=optim_kwargs['early_stop']['patience'])
            min_loss_output_psnr_histories = []
        
        optim_step_stats = None 
        with tqdm(range(
                optim_kwargs['iterations']), desc='DIP', disable=not show_pbar
            ) as pbar:
            for i in pbar:
                
                slicing_sequence = None
                if optim_kwargs['optim']['use_subsampling_orthospace']:
                    slicing_sequence = np.random.choice(range(
                        self.subspace.num_subspace_params), 
                        size=optim_kwargs['optim']['subsampling_orthospace_dim'],
                        replace=False, 
                        p=self.singular_probabilities.cpu().numpy()
                    )
                
                if optim_kwargs['optim']['optimizer'] == 'adam':

                    self.optimizer.zero_grad()
                    loss, output = self.objective(
                        criterion=criterion,
                        noisy_observation=noisy_observation,
                        use_tv_loss=use_tv_loss,
                        weight_decay=optim_kwargs['optim']['weight_decay'],
                        slicing_sequence=slicing_sequence,
                        gamma=optim_kwargs['optim']['gamma']
                        )
                    loss.backward()
                    self.optimizer.step()

                elif optim_kwargs['optim']['optimizer'] == 'lbfgs':

                    def closure():
                        self.optimizer.zero_grad()
                        loss, _ = self.objective(
                        criterion=criterion,
                        noisy_observation=noisy_observation,
                        use_tv_loss=use_tv_loss,
                        weight_decay=optim_kwargs['optim']['weight_decay'],
                        slicing_sequence=slicing_sequence,
                        gamma=optim_kwargs['optim']['gamma']
                        )
                        loss.backward(retain_graph=True)
                        return loss

                    loss = self.optimizer.step(closure)
                    output = self.forward(apply_forward_op=False)

                elif optim_kwargs['optim']['optimizer'] == 'ngd':
                    
                    self.optimizer.zero_grad()
                    partial_closure = partial(self.objective, 
                        criterion=criterion,
                        noisy_observation=noisy_observation,
                        use_tv_loss=use_tv_loss,
                        weight_decay=optim_kwargs['optim']['weight_decay'],
                        slicing_sequence=slicing_sequence,
                        gamma=optim_kwargs['optim']['gamma']
                    )

                    loss, output, optim_step_stats = self.optimizer.step(
                        closure=partial_closure,
                        curvature=fisher_info,
                        curvature_kwargs=curvature_update_kwargs,
                        hyperparams_kwargs=ngd_hyperparams_kwargs, 
                        use_adaptive_learning_rate=optim_kwargs['optim']['use_adaptive_learning_rate'],
                        use_adaptive_momentum=optim_kwargs['optim']['use_adaptive_momentum'],
                        use_adaptive_damping=optim_kwargs['optim']['use_adaptive_damping'],
                        use_approximate_quad_model=optim_kwargs['optim']['use_approximate_quad_model'],
                        return_stats=optim_kwargs['optim']['return_stats']
                    )
                else: 
                    raise NotImplementedError
                                
                if loss.item() < min_loss_state['loss']:
                    min_loss_state['loss'] = loss.item()
                    min_loss_state['output'] = output.detach()
                    min_loss_state['params_state_dict'] = deepcopy(self.subspace.state_dict())

                for p in self.nn_model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX
                
                if optim_step_stats is not None: 
                    stats_to_writer(optim_step_stats, writer, i)
        
                if ground_truth is not None:
                    min_loss_output_psnr = PSNR(
                            min_loss_state['output'].detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(
                            output.detach().cpu(), ground_truth.cpu())
                    pbar.set_description(f'DIP output_psnr={output_psnr:.3f}', refresh=False)
                    writer.add_scalar('min_loss_output_psnr', min_loss_output_psnr, i)
                    writer.add_scalar('output_psnr', output_psnr, i)
                    if optim_kwargs['early_stop']['use_early_stop']:
                        min_loss_output_psnr_histories.append(min_loss_output_psnr)

                writer.add_scalar('loss', loss.item(),  i)
                if i % 10 == 0:
                    writer.add_image('reco', normalize(
                            min_loss_state['output'][0, ...]).cpu().numpy(), i)

                if optim_kwargs['early_stop']['use_early_stop']:
                    #variance history
                    flat_out = output.detach().cpu().reshape(-1).numpy()
                    earlystop.update_img_collection(flat_out)
                    img_collection = earlystop.get_img_collection()
                    if len(img_collection) == optim_kwargs['early_stop']['buffer_size']:
                        ave_img = np.mean(img_collection, axis = 0)
                        variance = np.mean ( np.mean((np.stack(img_collection) - ave_img) **2, axis=1 ), axis=0)
                        writer.add_scalar('variance_early_stop', variance, i)
                        if earlystop.stop == False:
                            earlystop.stop = earlystop.check_stop(variance, i)
                        else:
                            writer.add_scalar('early_stop_detected', earlystop.best_epoch, earlystop.best_epoch)
                            writer.add_scalar('PSNR_at_early_stop', min_loss_output_psnr_histories[earlystop.best_epoch], earlystop.best_epoch)


        self.subspace.load_state_dict(
            min_loss_state['params_state_dict']
            )
        writer.close()

        return min_loss_state['output']
