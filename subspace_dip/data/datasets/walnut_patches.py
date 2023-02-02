"""
Provides the WalnutPatchesDataset.
"""
from typing import Tuple, Union, Iterator
import os
import numpy as np
import torch
import imageio
from torch import Tensor 
from sklearn.feature_extraction import image
from itertools import repeat

GT_NB_ITER = 50

def get_ground_truth(data_path, walnut_id, orbit_id, slice_ind):
    slice_path = os.path.join(
            data_path, 'Walnut{}'.format(walnut_id), 'Reconstructions',
            'full_AGD_{}_{:06}.tiff'.format(GT_NB_ITER, slice_ind))
    gt = imageio.imread(slice_path)

    return gt

class WalnutPatchesDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random patches of a Walnut.
    """
    def __init__(self, 
        data_path: str = './', 
        shape: Tuple[int, int] = (128, 128),
        max_patches: int = 32,
        walnut_id: int = 1,
        orbit_id: int = 2,
        slice_ind: int = 253,
        fixed_seed: int = 1
        ):
        super().__init__()

        self.shape = shape
        self.max_patches = max_patches
        self.walnut = get_ground_truth(data_path, walnut_id, orbit_id, slice_ind)[72:424, 72:424]
        self.fixed_seed = fixed_seed
        self.walnut_patches_data = []
        self.rng = np.random.RandomState(self.fixed_seed)
    

    def __len__(self) -> Union[int, float]:
        return self.max_patches if self.max_patches is not None else float('inf')
        
    def _extend_walnut_patches_data(self, min_length: int) -> None:

        n_to_generate = max(min_length - len(self.walnut_patches_data), 0)
        for _ in range(n_to_generate):
            patch = image.extract_patches_2d(
                self.walnut, 
                self.shape,
                max_patches=1,
                random_state=self.rng)
            self.walnut_patches_data.append(patch)

    def _generate_item(self, idx: int):
        image = self.walnut_patches_data[idx]
        return torch.from_numpy(image).float()

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.max_patches) if self.max_patches is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_walnut_patches_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_walnut_patches_data(idx + 1)
        return self._generate_item(idx)