from itertools import islice

import hydra
import torch

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from subspace_dip.dip import DeepImagePrior
from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_test_dataset
from subspace_dip.utils import PSNR, SSIM

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dtype = torch.get_default_dtype()
    device = torch.device(
        ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    if cfg.test_dataset.name in ['walnut']:
        dataset_kwargs_trafo = {
            'name': cfg.test_dataset.name, # proxy for disk_dist_ellipses
            'im_size': cfg.test_dataset.im_size,
            'data_path': cfg.test_dataset.data_path,
            'walnut_id': cfg.test_dataset.walnut_id
            }
    else:
        dataset_kwargs_trafo = {
            'name': cfg.test_dataset.name,
            'im_size': cfg.test_dataset.im_size
            }

    ray_trafo = get_standard_ray_trafo(
        ray_trafo_kwargs=OmegaConf.to_object(cfg.trafo),
        dataset_kwargs=dataset_kwargs_trafo,
    )    
    ray_trafo.to(dtype=dtype, device=device)

    # data: observation, ground_truth, filtbackproj
    net_kwargs = {
        'scales': cfg.dip.net.scales,
        'channels': cfg.dip.net.channels,
        'skip_channels': cfg.dip.net.skip_channels,
        'use_norm': cfg.dip.net.use_norm,
        'use_sigmoid': cfg.dip.net.use_sigmoid,
        'sigmoid_saturation_thresh': cfg.dip.net.sigmoid_saturation_thresh
    }

    reconstructor = DeepImagePrior(
        ray_trafo, 
        torch_manual_seed=cfg.dip.torch_manual_seed,
        device=device,
        net_kwargs=net_kwargs
    )

    dataset = get_standard_test_dataset(
        ray_trafo,
        dataset_kwargs=OmegaConf.to_object(cfg.test_dataset),
        trafo_kwargs=OmegaConf.to_object(cfg.trafo), 
        use_fixed_seeds_starting_from=cfg.seed,
        device=device, 
    )

    for i, data_sample in enumerate(islice(DataLoader(dataset), cfg.num_images)):

        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate
        
        observation, ground_truth, filtbackproj = data_sample
        
        observation = observation.to(dtype=dtype, device=device)
        filtbackproj = filtbackproj.to(dtype=dtype, device=device)
        ground_truth = ground_truth.to(dtype=dtype, device=device)

        use_norm_op = cfg.trafo.get('use_norm_op', False)
        gamma = cfg.dip.optim.gamma if not use_norm_op else cfg.dip.optim.gamma * ray_trafo.norm_const **2

        optim_kwargs = {
                'lr': cfg.dip.optim.lr,
                'iterations': cfg.dip.optim.iterations,
                'loss_function': cfg.dip.optim.loss_function,
                'gamma': gamma,
                'use_early_stop': cfg.dip.optim.use_early_stop,
                'buffer_size': cfg.dip.optim.buffer_size,
                'patience': cfg.dip.optim.patience
                }
        
        if cfg.load_dip_models_from_path is not None: 
            reconstructor.load_pretrain_model(
                learned_params_path=cfg.load_dip_models_from_path
            )
        else: 
            reconstructor.init_nn_model(
                torch_manual_seed=cfg.seed
            )

        recon = reconstructor.reconstruct(
            noisy_observation = observation,
            filtbackproj=filtbackproj,
            ground_truth=ground_truth,
            recon_from_randn=cfg.dip.recon_from_randn,
            log_path=cfg.dip.log_path,
            optim_kwargs=optim_kwargs
        )

        torch.save(reconstructor.nn_model.state_dict(),
                './dip_model_{}.pt'.format(i))

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()
