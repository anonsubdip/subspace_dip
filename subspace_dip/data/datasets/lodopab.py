"""
Provides the LoDoPaBTorchDataset.
"""
from typing import Iterator
import numpy as np
import torch 
from torch import Tensor
from odl import uniform_discr
from ..simulation import SimulatedDataset
from subspace_dip.data.trafo.base_ray_trafo import BaseRayTrafo
try:
    from dival.datasets import LoDoPaBDataset
except ImportError:
    LoDoPaBDataset = None

class LoDoPaBTorchDataset(torch.utils.data.IterableDataset):
    """
    Dataset with ground truth images from LoDoPaB.
    """
    def __init__(self,
            fold : str = 'train'
        ):
        
        self.fold = fold
        self.shape = (362, 362)
        min_pt = (-0.13, -0.13)
        max_pt = (0.13, 0.13)
        self.space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.length = {'train': 35820, 'validation': 3522, 'test': 3553}[fold]
        self.lodopab = None  # set up on demand to avoid errors when not actually using data
        super().__init__()

    def __len__(self) -> int:
        return self.length

    def _setup_lodopab(self) -> None:
        if self.lodopab is None:
            self.lodopab = LoDoPaBDataset(impl='astra_cpu')  # impl does not matter since we only use the images

    def __iter__(self) -> Iterator[Tensor]:
        assert LoDoPaBDataset is not None, 'dival.datasets.lodopab.LoDoPaBDataset could not be imported, but is required by LoDoPaBDataset'
        self._setup_lodopab()
        for i in range(self.length):
            yield torch.from_numpy(self.lodopab.get_sample(i, part=self.fold, out=(False, True))[1].asarray()).unsqueeze(dim=0)

    def __getitem__(self, idx: int) -> Tensor:
        self._setup_lodopab()
        return torch.from_numpy(self.lodopab.get_sample(idx, part=self.fold, out=(False, True))[1].asarray()).unsqueeze(dim=0)

def get_lodopab_dataset(
        ray_trafo: BaseRayTrafo,
        fold : str = 'train',
        white_noise_rel_stddev : float = .05,
        use_fixed_seeds_starting_from : int = 1, 
        device = None) -> SimulatedDataset:

    image_dataset = LoDoPaBTorchDataset(
            fold=fold, 
            )

    return SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=white_noise_rel_stddev,
            use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
            device=device
        )
