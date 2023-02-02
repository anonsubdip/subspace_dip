"""
Provides :class:`DeepImagePrior`.
"""
from typing import Optional, Union
from warnings import warn
from copy import deepcopy

import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX

from torch import Tensor
from torch.nn import MSELoss
from tqdm import tqdm

from .base_dip_image_prior import BaseDeepImagePrior
from .early_stopping_criteria import EarlyStop
from subspace_dip.utils import tv_loss, PSNR, normalize
from subspace_dip.data import BaseRayTrafo


class DeepImagePrior(BaseDeepImagePrior):

    def __init__(self,
            ray_trafo: BaseRayTrafo,
            torch_manual_seed: Union[int, None] = 1,
            device=None,
            net_kwargs=None):
        
        super().__init__(
            ray_trafo=ray_trafo,
            torch_manual_seed=torch_manual_seed,
            device=device,
            net_kwargs=net_kwargs
        )

    def reconstruct(self,
            noisy_observation: Tensor,
            filtbackproj: Optional[Tensor] = None,
            ground_truth: Optional[Tensor] = None,
            recon_from_randn: bool = False,
            use_tv_loss: bool = True,
            log_path: str = '.',
            show_pbar: bool = True,
            optim_kwargs=None) -> Tensor:

        writer = tensorboardX.SummaryWriter(
                logdir=os.path.join(log_path, '_'.join((
                        datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        socket.gethostname(),
                        '_DIP' if not use_tv_loss else '_DIP+TV'))))

        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('gamma', 1e-4)
        optim_kwargs.setdefault('lr', 1e-4)
        optim_kwargs.setdefault('iterations', 10000)
        optim_kwargs.setdefault('loss_function', 'mse')

        self.nn_model.train()

        self.net_input = (
            0.1 * torch.randn(1, 1, *self.ray_trafo.im_shape, device=self.device)
            if recon_from_randn else
            filtbackproj.to(self.device))

        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=optim_kwargs['lr'])
        noisy_observation = noisy_observation.to(self.device)
        if optim_kwargs['loss_function'] == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        min_loss_state = {
            'loss': np.inf,
            'output': self.nn_model(self.net_input).detach(),  # pylint: disable=not-callable
            'params_state_dict': deepcopy(self.nn_model.state_dict()),
        }
       
        writer.add_image('filtbackproj', normalize(
               filtbackproj[0, ...]).cpu().numpy(), 0)
        if ground_truth is not None:
            writer.add_image('ground_truth', normalize(
                ground_truth[0, ...]).cpu().numpy(), 0)

        if optim_kwargs['use_early_stop']: 
            earlystop = EarlyStop(size=optim_kwargs['buffer_size'], patience=optim_kwargs['patience'])
            min_loss_output_psnr_histories = []

        with tqdm(range(optim_kwargs['iterations']), desc='DIP', disable=not show_pbar) as pbar:

            for i in pbar:
                self.optimizer.zero_grad()
                output = self.nn_model(self.net_input)  # pylint: disable=not-callable
                loss = criterion(self.ray_trafo(output), noisy_observation)
                if use_tv_loss:
                    loss = loss + optim_kwargs['gamma'] * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1)

                if loss.item() < min_loss_state['loss']:
                    min_loss_state['loss'] = loss.item()
                    min_loss_state['output'] = output.detach()
                    min_loss_state['params_state_dict'] = deepcopy(self.nn_model.state_dict())
                
                self.optimizer.step()

                for p in self.nn_model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if ground_truth is not None:
                    min_loss_output_psnr = PSNR(
                            min_loss_state['output'].detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(
                            output.detach().cpu(), ground_truth.cpu())
                    pbar.set_description(f'DIP output_psnr={output_psnr:.1f}', refresh=False)
                    writer.add_scalar('min_loss_output_psnr', min_loss_output_psnr, i)
                    writer.add_scalar('output_psnr', output_psnr, i)
                    if optim_kwargs['use_early_stop']:
                        min_loss_output_psnr_histories.append(min_loss_output_psnr)

                writer.add_scalar('loss', loss.item(),  i)
                if i % 10 == 0:
                    writer.add_image('reco', normalize(
                            min_loss_state['output'][0, ...]).cpu().numpy(), i)
                
                if optim_kwargs['use_early_stop']:
                    #variance history
                    flat_out = output.detach().cpu().reshape(-1).numpy()
                    earlystop.update_img_collection(flat_out)
                    img_collection = earlystop.get_img_collection()
                    if len(img_collection) == optim_kwargs['buffer_size']:
                        ave_img = np.mean(img_collection, axis = 0)
                        variance = np.mean ( np.mean((np.stack(img_collection) - ave_img) **2, axis=1 ), axis=0)
                        writer.add_scalar('variance_early_stop', variance, i)
                        if earlystop.stop == False:
                            earlystop.stop = earlystop.check_stop(variance, i)
                        else:
                            writer.add_scalar('early_stop_detected', earlystop.best_epoch, earlystop.best_epoch)
                            writer.add_scalar('PSNR_at_early_stop', min_loss_output_psnr_histories[earlystop.best_epoch], earlystop.best_epoch)

        self.nn_model.load_state_dict(min_loss_state['params_state_dict'])
        writer.close()

        return min_loss_state['output']
