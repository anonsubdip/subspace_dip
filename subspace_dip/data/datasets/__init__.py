"""
Provides synthetic image datasets and access to stored datasets.
"""

from .rectangles import RectanglesDataset
from .ellipses import EllipsesDataset, DiskDistributedEllipsesDataset, get_ellipses_dataset, get_disk_dist_ellipses_dataset
from .lodopab import LoDoPaBTorchDataset, get_lodopab_dataset
from .walnut_patches import WalnutPatchesDataset
from .cartoonset import CartoonSetDataset
from .walnut import get_walnut_2d_observation, get_walnut_2d_ground_truth
from .mayo import MayoDataset