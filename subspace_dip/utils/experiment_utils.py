from typing import Dict, Optional, List

import os
import numpy as np

from torch.utils.data import Dataset, TensorDataset
from subspace_dip.data import get_ray_trafo, SimulatedDataset
from subspace_dip.data import (
        RectanglesDataset, EllipsesDataset, WalnutPatchesDataset, 
        CartoonSetDataset, MayoDataset, get_walnut_2d_observation, get_walnut_2d_ground_truth
    )
from .utils import get_original_cwd

def get_standard_ray_trafo(ray_trafo_kwargs: dict, dataset_kwargs: Dict):
    kwargs = {}
    kwargs['angular_sub_sampling'] = ray_trafo_kwargs['angular_sub_sampling']
    if dataset_kwargs['name'] in ('ellipses', 'rectangles', 'walnut_patches', 'cartoonset'):
        kwargs['im_shape'] = (dataset_kwargs['im_size'], dataset_kwargs['im_size'])
        kwargs['num_angles'] = ray_trafo_kwargs['num_angles']
    elif dataset_kwargs['name'] in ('mayo', 'ellipses_mayo', 'lodopab_mayo_cropped', 'mayo_cropped'): 
        kwargs['im_shape'] = (dataset_kwargs['im_size'], dataset_kwargs['im_size'])
        kwargs['num_angles'] = ray_trafo_kwargs['num_angles']
        kwargs['src_radius'] = ray_trafo_kwargs['src_radius']
        kwargs['det_radius'] = ray_trafo_kwargs['det_radius']
        kwargs['use_norm_op'] = ray_trafo_kwargs['use_norm_op']
        # kwargs['load_mat_from_path'] = ray_trafo_kwargs['load_mat_from_path']
    elif dataset_kwargs['name'] in ('walnut', ):
        kwargs['data_path'] = os.path.join(get_original_cwd(), dataset_kwargs['data_path'])
        kwargs['matrix_path'] = os.path.join(get_original_cwd(), dataset_kwargs['data_path'])
        kwargs['walnut_id'] = dataset_kwargs['walnut_id']
        kwargs['orbit_id'] = ray_trafo_kwargs['orbit_id']
        kwargs['proj_col_sub_sampling'] = ray_trafo_kwargs['proj_col_sub_sampling']
    else:
        raise ValueError
    return get_ray_trafo(dataset_kwargs['name'], kwargs=kwargs)

def get_standard_test_dataset(
        ray_trafo,
        dataset_kwargs,
        trafo_kwargs, 
        use_fixed_seeds_starting_from=1, 
        device=None
    ) -> Dataset:

    if dataset_kwargs['name'] == 'ellipses':

        image_dataset = EllipsesDataset(
                (dataset_kwargs['im_size'], dataset_kwargs['im_size']), 
                length=dataset_kwargs['length']['test'],
                )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    
    elif dataset_kwargs['name'] == 'rectangles':

        image_dataset = RectanglesDataset(
                (dataset_kwargs['im_size'], dataset_kwargs['im_size']),
                num_rects=dataset_kwargs['num_rects'],
                num_angle_modes=dataset_kwargs['num_angle_modes'],
                angle_modes_sigma=dataset_kwargs['angle_modes_sigma'])
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    
    elif dataset_kwargs['name'] == 'walnut_patches':

        image_dataset = WalnutPatchesDataset(
            data_path=dataset_kwargs['data_path_test'], shape=(
                dataset_kwargs['im_size'], dataset_kwargs['im_size']
                ),
            walnut_id=dataset_kwargs['walnut_id'], orbit_id=dataset_kwargs['orbit_id'], 
            slice_ind=dataset_kwargs['slice_ind'], 
            )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif dataset_kwargs['name'] == 'cartoonset':

        image_dataset = CartoonSetDataset(
            data_path=dataset_kwargs['data_path_test'], shape=(
                dataset_kwargs['im_size'], dataset_kwargs['im_size']
                )
            )
        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)
    
    elif dataset_kwargs['name'] in ('mayo', 'mayo_cropped'):

        image_dataset = MayoDataset(
            data_path=dataset_kwargs['data_path'], sample_names=dataset_kwargs['sample_names'],
            shape=(dataset_kwargs['im_size'], dataset_kwargs['im_size']), num_slice_per_patient=dataset_kwargs['num_slice_per_patient'], 
            crop=(dataset_kwargs['name'] == 'mayo_cropped'),
            seed=use_fixed_seeds_starting_from
        )

        dataset = SimulatedDataset(
                image_dataset, ray_trafo,
                white_noise_rel_stddev=dataset_kwargs['noise_stddev'],
                use_fixed_seeds_starting_from=use_fixed_seeds_starting_from,
                device=device)

    elif dataset_kwargs['name'] == 'walnut':

        noisy_observation = get_walnut_2d_observation(
                data_path=os.path.join(get_original_cwd(),dataset_kwargs['data_path']),
                walnut_id=dataset_kwargs['walnut_id'], orbit_id=trafo_kwargs['orbit_id'],
                angular_sub_sampling=trafo_kwargs['angular_sub_sampling'],
                proj_col_sub_sampling=trafo_kwargs['proj_col_sub_sampling'],
                scaling_factor=dataset_kwargs['scaling_factor']).to(device=device)
        ground_truth = get_walnut_2d_ground_truth(
                data_path=os.path.join(get_original_cwd(),dataset_kwargs['data_path']),
                walnut_id=dataset_kwargs['walnut_id'], orbit_id=trafo_kwargs['orbit_id'],
                scaling_factor=dataset_kwargs['scaling_factor']).to(device=device)
        filtbackproj = ray_trafo.fbp(
                noisy_observation[None].to(device=device))[0].to(device=device)
        dataset = TensorDataset(  # include batch dims
                noisy_observation[None], ground_truth[None], filtbackproj[None])
    else:
        raise ValueError

    return dataset

def find_log_files(log_dir: str) -> str:
    log_files = []
    for path, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                log_files.append(os.path.join(path, file))
    if not log_files:
        raise RuntimeError(f'did not find log file in {log_dir}')
    return log_files

def extract_tensorboard_scalars(
        log_file: str, save_as_npz: str = '', tags: Optional[List[str]] = None) -> dict:
    """
    From https://github.com/educating-dip/bayes_dip/blob/5ae7946756d938a7cd00ad56307a934b8dd3685e/bayes_dip/utils/evaluation_utils.py#L693
    Extract scalars from a tensorboard log file.
    Parameters
    ----------
    log_file : str
        Tensorboard log filepath.
    save_as_npz : str, optional
        File path to save the extracted scalars as a npz file.
    tags : list of str, optional
        If specified, only extract these tags.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError:
        raise RuntimeError('Tensorboard\'s event_accumulator could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

    ea = event_accumulator.EventAccumulator(
            log_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    tags = tags or ea.Tags()['scalars']

    scalars = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        times = [event.wall_time for event in events]
        scalars[tag + '_steps'] = np.asarray(steps)
        scalars[tag + '_scalars'] = np.asarray(values)
        scalars[tag + '_times'] = np.asarray(times)

    if save_as_npz:
        np.savez(save_as_npz, **scalars)

    return scalars

def print_dct(dct):
    for (item, values) in dct.items():
        print(item)
        for value in values:
            print(value)

# from https://stackoverflow.com/a/47882384
def sorted_dict(d):
    return {k: sorted_dict(v) if isinstance(v, dict) else v
            for k, v in sorted(d.items())}
