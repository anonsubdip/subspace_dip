"""
Provides :class:`BaseDeepImagePrior`.
"""
from typing import Union
from abc import ABC, abstractmethod
from contextlib import nullcontext

import os
import torch

from .network import UNet
from subspace_dip.utils import get_original_cwd
from subspace_dip.data import BaseRayTrafo

class BaseDeepImagePrior(ABC):
    """
    Base class for CT reconstructor applying DIP with TV regularization (see [2]_).
    The DIP was introduced in [1].
    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018, "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           https://doi.org/10.1109/CVPR.2018.00984
    .. [2] D. Otero Baguer, J. Leuschner, M. Schmidt, 2020, "Computed
           Tomography Reconstruction Using Deep Image Prior and Learned
           Reconstruction Methods". Inverse Problems.
           https://doi.org/10.1088/1361-6420/aba415
    """

    def __init__(self,
            ray_trafo: BaseRayTrafo,
            torch_manual_seed: Union[int, None] = 1,
            device=None,
            net_kwargs=None):

        self.device = device or torch.device(
                ('cuda:0' if torch.cuda.is_available() else 'cpu')
            )
        self.ray_trafo = ray_trafo.to(self.device)
        self.net_kwargs = net_kwargs
        self.init_nn_model(torch_manual_seed)
        self.net_input = None
        self.optimizer = None

    def init_nn_model(self,
            torch_manual_seed: Union[int, None]):

        with (torch.random.fork_rng([self.device]) if torch_manual_seed is not None
                else nullcontext()):
            if torch_manual_seed is not None:
                torch.random.manual_seed(torch_manual_seed)

            self.nn_model = UNet(
                in_ch=1,
                out_ch=1,
                channels=self.net_kwargs['channels'][:self.net_kwargs['scales']],
                skip_channels=self.net_kwargs['skip_channels'][:self.net_kwargs['scales']],
                use_sigmoid=self.net_kwargs['use_sigmoid'],
                use_norm=self.net_kwargs['use_norm'],
                sigmoid_saturation_thresh= self.net_kwargs['sigmoid_saturation_thresh']
                ).to(self.device)

    def load_pretrain_model(self,
            learned_params_path: str):

        path = os.path.join(
            get_original_cwd(),
            learned_params_path if learned_params_path.endswith('.pt') \
                else learned_params_path + '.pt')
        self.nn_model.load_state_dict(torch.load(path, map_location=self.device))
    
    @abstractmethod
    def reconstruct(self, ) -> None:
        raise NotImplementedError