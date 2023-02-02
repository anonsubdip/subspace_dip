from omegaconf import DictConfig, OmegaConf

import torch
import hydra

from subspace_dip.utils.experiment_utils import get_standard_ray_trafo
from subspace_dip.dip import DeepImagePrior, ParameterSampler, LinearSubspace

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dtype = torch.get_default_dtype()
    device = torch.device(
        ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    if cfg.test_dataset.name in ['walnut']:
        dataset_kwargs_trafo = {
            'name': cfg.test_dataset.name,
            'im_size': cfg.test_dataset.im_size,
            'data_path': cfg.test_dataset.data_path,
            'walnut_id': cfg.test_dataset.walnut_id
            }
    else:
        dataset_kwargs_trafo = {
            'name': cfg.test_dataset.name,
            'im_size': cfg.test_dataset.im_size
            }

    ray_trafo = get_standard_ray_trafo( # placeholder 
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
                ray_trafo, # placeholder 
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

    sampler.load_sampled_paramters(
        path_to_parameters_samples=cfg.sampler.path_to_parameters_samples
    )

    subspace = LinearSubspace(
        parameters_samples_list=sampler.parameters_samples, 
        subspace_dim=cfg.subspace.subspace_dim,
        use_random_init=cfg.subspace.use_random_init,
        num_random_projs=cfg.subspace.num_random_projs,
        use_approx=cfg.subspace.use_approx,
        device=device
    )
    
    subspace.save_ortho_basis(
        ortho_basis_path=cfg.subspace.ortho_basis_path
        )

if __name__ == '__main__':
    coordinator()
