"""
Provides the CartoonSetDataset.
"""
from typing import Tuple, Union, Iterator, List
import os
import torch
import numpy as np
from pydicom import dcmread
from torch import Tensor
from itertools import repeat
from PIL import Image, ImageOps
from torchvision import transforms

def get_paths_and_dcm_files(sample_names: List[str], data_path: str = './', num_slice_per_patient: int = 20):
    
    dirs = [os.listdir(os.path.join(data_path, sample_name)) for sample_name in sample_names]
    assert all (len(d) == 1 for d in dirs)
    full_dose_images_dirs = []
    for sample_name, dirr in zip(sample_names, dirs):
        full_dose_images_dirs.append( 
            [d for d in os.listdir(os.path.join(data_path, sample_name, dirr[0])) if 'Full Dose Images' in d]
        )
    assert all ( len(full_dose_images_dir) == 1 for full_dose_images_dir in full_dose_images_dirs)
    paths = []
    for sample_name, dirr, full_dose_images_dir in zip(sample_names, dirs, full_dose_images_dirs):
        paths.append(os.path.join(data_path, sample_name, dirr[-1], full_dose_images_dir[-1]))
    dcm_files = [os.listdir(path) for path in paths]
    paths_to_flat_dcm_files = []
    for dcm_file, path in zip(dcm_files, paths):
        dcm_file.sort(
                key=lambda f: float(dcmread(os.path.join(path, f), 
                specific_tags=['SliceLocation'])['SliceLocation'].value)
            )
        half_bin_width = 0.5 / num_slice_per_patient * len(dcm_file)
        idx = np.round(np.linspace(half_bin_width, len(dcm_file) - half_bin_width - 1, num_slice_per_patient)).astype(int)
        paths_to_flat_dcm_files.extend([os.path.join(path, dcm_file[i]) for i in idx])
    return paths_to_flat_dcm_files

def mayo_to_tensor(
    path: str = './',
    shape: Tuple[int, int] = (512, 512),
    crop: bool = False,
    seed: int = 1,
    ):

    transform = transforms.Compose(
            [transforms.ToTensor(), (transforms.CenterCrop(shape) if crop else transforms.Resize(shape))]
        )
    dcm_image = dcmread(path)
    image = dcm_image.pixel_array.astype('float32').T
    rng = np.random.default_rng((seed + hash(path)) % (2**64))

    # rescale by dicom meta info
    image *= dcm_image.RescaleSlope
    image += dcm_image.RescaleIntercept
    image += rng.uniform(0., 1., size=image.shape)
    # convert values
    MU_WATER = 20
    MU_AIR = 0.02
    MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    image *= (MU_WATER - MU_AIR) / 1000
    image += MU_WATER
    image /= MU_MAX
    np.clip(image, 0., 1., out=image)
    return transform(image).numpy()

class MayoDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random rectangles.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self,
        sample_names: List[str],
        data_path: str = './',
        num_slice_per_patient: int = 20,
        shape: Tuple[int, int] = (512, 512),
        crop: bool = False,
        seed: int = 1, 
        ):

        super().__init__()
        self.shape = shape
        self.crop = crop
        paths_to_flat_dcm_files = get_paths_and_dcm_files(
                sample_names=sample_names,
                data_path=data_path,
                num_slice_per_patient=num_slice_per_patient
            )
        self.paths_to_flat_dcm_files = iter(paths_to_flat_dcm_files)
        self.length = len(paths_to_flat_dcm_files)
        self.seed = seed
        self.mayo_data = []
        
    def __len__(self) -> Union[int, float]:
        return self.length if self.length is not None else float('inf')
        
    def _extend_mayo_data(self, min_length: int) -> None:

        n_to_generate = max(min_length - len(self.mayo_data), 0)
        for _ in range(n_to_generate):
            path = next(self.paths_to_flat_dcm_files)
            scan_slice = mayo_to_tensor(
                    path=path,
                    shape=self.shape,
                    crop=self.crop,
                    seed=self.seed
                )
            self.mayo_data.append(scan_slice)

    def _generate_item(self, idx: int):
        image = self.mayo_data[idx]
        return torch.from_numpy(image).float()

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_mayo_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_mayo_data(idx + 1)
        return self._generate_item(idx)