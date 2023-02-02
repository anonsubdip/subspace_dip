from omegaconf import DictConfig, OmegaConf

import torch
import hydra

from subspace_dip.utils.experiment_utils import get_standard_ray_trafo
from subspace_dip.dip import DeepImagePrior, ParameterSampler

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dtype = torch.get_default_dtype()
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    assert cfg.source_dataset.im_size == cfg.test_dataset.im_size
    if cfg.test_dataset.name in ['walnut']:
        dataset_kwargs_trafo = {
            'name': cfg.test_dataset.name,
            'im_size': cfg.source_dataset.im_size, 
            'data_path': cfg.test_dataset.data_path,
            'walnut_id': cfg.test_dataset.walnut_id
            }
    else:
        dataset_kwargs_trafo = {
            'name': cfg.source_dataset.name,
            'im_size': cfg.source_dataset.im_size 
            }

    ray_trafo = get_standard_ray_trafo(
        ray_trafo_kwargs=OmegaConf.to_object(cfg.trafo), 
        dataset_kwargs=dataset_kwargs_trafo
    )
    ray_trafo.to(dtype=dtype, device=device)

    net_kwargs = {
            'scales': cfg.dip.net.scales,
            'channels': cfg.dip.net.channels,
            'skip_channels': cfg.dip.net.skip_channels,
            'use_norm': cfg.dip.net.use_norm,
            'use_sigmoid': cfg.dip.net.use_sigmoid,
            'sigmoid_saturation_thresh': cfg.dip.net.sigmoid_saturation_thresh
        }

    base_reconstructor = DeepImagePrior(
        ray_trafo, 
        torch_manual_seed=cfg.dip.torch_manual_seed,
        device=device, 
        net_kwargs=net_kwargs
        )
    
    if cfg.load_dip_models_from_path is not None: 
        base_reconstructor.load_pretrain_model(
            learned_params_path=cfg.load_dip_models_from_path)
            
    sampler = ParameterSampler(
            model=base_reconstructor.nn_model,
            device=device
        )

    optim_kwargs = {
        'log_path': './',
        'seed': cfg.seed, 
        'torch_manual_seed': cfg.dip.torch_manual_seed,
        'save_best_learned_params_path': './',
        'save_best_learned_params_per_epoch': cfg.sampler.training.save_best_learned_params_per_epoch,
        'epochs': cfg.sampler.training.num_epochs,
        'num_samples': cfg.sampler.sampling.num_samples,
        'burn_in': cfg.sampler.sampling.burn_in,
        'sampling_strategy': cfg.sampler.sampling.sampling_strategy,
        'batch_size': cfg.sampler.training.batch_size,
        'optimizer': {
            'lr': cfg.sampler.training.optim.lr,
            'weight_decay': cfg.sampler.training.optim.weight_decay,
        },
        'scheduler': {
            'name': cfg.sampler.training.optim.scheduler_name,
            'train_len': cfg.source_dataset.length,
            'lr_min': cfg.sampler.training.optim.lr_min,
            'max_lr': cfg.sampler.training.optim.max_lr
        }
    }

    if cfg.source_dataset.name in ('ellipses', 'ellipses_mayo'): 
        dataset_kwargs = {
            'name': cfg.source_dataset.name,
            'im_size': cfg.source_dataset.im_size,
            'length':{
                'train': cfg.source_dataset.length.train,
                'validation': cfg.source_dataset.length.validation,
                },
            'max_n_ellipse': cfg.source_dataset.max_n_ellipse,
            'white_noise_rel_stddev': cfg.source_dataset.noise_stddev,
            'use_fixed_seeds_starting_from': cfg.seed, 
        }

    elif cfg.source_dataset.name == 'disk_dist_ellipses':
        dataset_kwargs = {
            'name': cfg.source_dataset.name,
            'im_size': cfg.source_dataset.im_size,
            'diameter': cfg.source_dataset.diameter,
            'length':{
                'train': cfg.source_dataset.length.train,
                'validation': cfg.source_dataset.length.validation,
                },
            'white_noise_rel_stddev': cfg.source_dataset.noise_stddev,
            'use_fixed_seeds_starting_from': cfg.seed,
        }
    elif cfg.source_dataset.name in ('lodopab_mayo_cropped', ): 
        dataset_kwargs = {
            'name': cfg.source_dataset.name,
            'white_noise_rel_stddev': cfg.source_dataset.noise_stddev,
            'use_fixed_seeds_starting_from': cfg.seed, 
        }
    
    else: 
        raise  NotImplementedError

    sampler.sample(
        ray_trafo=ray_trafo,
        dataset_kwargs=dataset_kwargs, 
        optim_kwargs=optim_kwargs, 
        save_samples=cfg.sampler.sampling.save_samples
    )

if __name__ == '__main__':
    coordinator()
