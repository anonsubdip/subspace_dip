"""
Provides the CartoonSetDataset.
"""
from typing import Tuple, Union, Iterator
import os
import glob
import torch
from torch import Tensor
from itertools import repeat
from PIL import Image, ImageOps
from torchvision import transforms

def ground_truth_images_paths(data_path: str = './'):
    # paths = glob.glob(
    #     os.path.join(data_path, '*.png')
    #     )
    paths = ['cs11385766627408421941.png', 'cs10163407696237708017.png', 'cs10789964898752667468.png', 'cs11432782836976689768.png', 'cs10776951687697745429.png', 'cs10643897309345068956.png', 'cs11007836227924175225.png', 'cs11047606218939922009.png', 'cs10673412480564654371.png', 'cs10467585218834139847.png', 'cs10482383848225144714.png', 'cs11472999992588988642.png', 'cs10962370497078781544.png', 'cs1016960315622698178.png', 'cs11094869622560834633.png', 'cs10815329220610882121.png', 'cs10286443993893124999.png', 'cs10945133897810646556.png', 'cs10027877343301002937.png', 'cs11226866211118352735.png', 'cs10355407661266895133.png', 'cs11150006532861639274.png', 'cs10873564502044053704.png', 'cs10055084037892353601.png', 'cs11448867488293195306.png', 'cs10702042775381567948.png', 'cs11371496047779617821.png', 'cs11206312379670862496.png', 'cs10202230117408697473.png', 'cs10110500133085304289.png', 'cs1076624962556372155.png', 'cs10652123026070622682.png', 'cs10984353270512289716.png', 'cs10725880427456484228.png', 'cs11437552831834952704.png', 'cs1122915133441396301.png', 'cs10256269974907253625.png', 'cs10138029120411248971.png', 'cs10163208946417580129.png', 'cs11234788610139072238.png', 'cs10704187086790035420.png', 'cs10007045796666045181.png', 'cs11073898730996325022.png', 'cs11401539041143189220.png', 'cs10516486517001935066.png', 'cs11151312874489582225.png', 'cs11405895699509697365.png', 'cs10358512809133733013.png', 'cs10179040591972554298.png', 'cs10321800737649479180.png', 'cs10803209600551526135.png', 'cs11236128051640659150.png', 'cs10006738088178033432.png', 'cs10735306698098094685.png', 'cs1065290654929907116.png', 'cs10942906195141459505.png', 'cs1121932471030234669.png', 'cs11370492422015378521.png', 'cs11094968404010812457.png', 'cs10922238697452833340.png']
    paths = [os.path.join(data_path, s) for s in paths]
    return iter(paths), len(paths)

def cartoon_to_tensor(image_path: str = './', 
    shape: Tuple[int, int] = (128, 128),
    ):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(
                    num_output_channels=1
                ),
            transforms.Resize(shape)]
        )
    return transform(
        ImageOps.invert(Image.open(image_path).convert('RGB'))
        ).numpy()

class CartoonSetDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random rectangles.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, 
        data_path: str = './',
        shape: Tuple[int, int] = (128, 128),
        ):

        super().__init__()
        self.shape = shape
        self.cartoonset, self.length = ground_truth_images_paths(
                data_path=data_path
            )
        self.cartoonset_data = []
        
    def __len__(self) -> Union[int, float]:
        return self.length if self.length is not None else float('inf')
        
    def _extend_cartoonset_data(self, min_length: int) -> None:

        n_to_generate = max(min_length - len(self.cartoonset_data), 0)
        for _ in range(n_to_generate):
            image_path = next(self.cartoonset)
            cartoon = cartoon_to_tensor(
                    image_path=image_path,
                    shape=self.shape
                )
            self.cartoonset_data.append(cartoon)

    def _generate_item(self, idx: int):
        image = self.cartoonset_data[idx]
        return torch.from_numpy(image).float()

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_cartoonset_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_cartoonset_data(idx + 1)
        return self._generate_item(idx)