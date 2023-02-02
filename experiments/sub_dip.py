from itertools import islice
from omegaconf import DictConfig, OmegaConf

import hydra
import torch

from torch.utils.data import DataLoader

from subspace_dip.dip import DeepImagePrior, SubspaceDeepImagePrior, LinearSubspace, FisherInfo
from subspace_dip.utils import PSNR, SSIM
from subspace_dip.utils.experiment_utils import get_standard_ray_trafo, get_standard_test_dataset


@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dtype = torch.get_default_dtype()
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    assert cfg.test_dataset.im_size == cfg.source_dataset.im_size

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
    
    base_reconstructor.load_pretrain_model(
        learned_params_path=cfg.load_dip_models_from_path)

    subspace = LinearSubspace(
        subspace_dim=cfg.subspace.subspace_dim,
        use_random_init=cfg.subspace.use_random_init,
        num_random_projs=cfg.subspace.num_random_projs,
        load_ortho_basis_path=cfg.subspace.ortho_basis_path,
        params_space_retain_ftc=cfg.subspace.params_space_retain_ftc,
        device=device
    )

    reconstructor = SubspaceDeepImagePrior(
        ray_trafo=ray_trafo,
        subspace=subspace,
        state_dict=base_reconstructor.nn_model.state_dict(),
        torch_manual_seed=cfg.dip.torch_manual_seed,
        device=device, 
        net_kwargs=net_kwargs
    )

    fisher_info = None
    if cfg.subspace.fine_tuning.optim.optimizer == 'ngd':
        
        fisher_info = FisherInfo(
            subspace_dip=reconstructor,
            init_damping=cfg.subspace.fisher_info.init_damping,
            init_curvature_ema=cfg.subspace.fisher_info.static_curvature_ema,
            sampling_probes_mode=cfg.subspace.fisher_info.sampling_probes_mode
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
        gamma = cfg.subspace.fine_tuning.optim.gamma if not use_norm_op else cfg.subspace.fine_tuning.optim.gamma * ray_trafo.norm_const **2

        optim_kwargs = { # optim_kwargs for adam and lbfgs
            'iterations': cfg.subspace.fine_tuning.iterations,
            'loss_function': cfg.subspace.fine_tuning.loss_function,
            'early_stop': {
                'use_early_stop': cfg.dip.optim.use_early_stop,
                'buffer_size': cfg.dip.optim.buffer_size,
                'patience': cfg.dip.optim.patience
                },
            'optim':{
                    'lr': cfg.subspace.fine_tuning.optim.lr,
                    'optimizer': cfg.subspace.fine_tuning.optim.optimizer,
                    'gamma': gamma,
                    'weight_decay': cfg.subspace.fine_tuning.optim.weight_decay,
                    'use_subsampling_orthospace': cfg.subspace.use_subsampling_orthospace,
                    'subsampling_orthospace_dim': cfg.subspace.subsampling_orthospace.subsampling_orthospace_dim
                }
        }

        if cfg.subspace.fine_tuning.optim.optimizer == 'ngd': 
            optim_kwargs['optim'].update(
                { # optim_kwargs specific to ngd
                    'use_adaptive_damping': cfg.subspace.fine_tuning.optim.use_adaptive_damping,
                    'use_adaptive_learning_rate': cfg.subspace.fine_tuning.optim.use_adaptive_learning_rate,
                    'use_adaptive_momentum': cfg.subspace.fine_tuning.optim.use_adaptive_momentum,
                    'momentum': cfg.subspace.fine_tuning.optim.momentum,
                    'init_scale_curvature': cfg.subspace.fine_tuning.optim.init_scale_curvature,
                    'use_approximate_quad_model': cfg.subspace.fine_tuning.optim.use_approximate_quad_model,
                    'stats_interval': cfg.subspace.fine_tuning.optim.stats_interval,
                    'return_stats': cfg.subspace.fisher_info.return_stats,
                    # optim_kwargs specific to NGD hyperparams (max/min-lr, min-momentum, etc)
                    'hyperparams_kwargs': OmegaConf.to_object(cfg.subspace.fine_tuning.hyperparams_kwargs),
                    # optim_kwargs specific to FisherInfo update
                    'num_random_vecs': cfg.subspace.fisher_info.num_random_vecs,
                    'mode': cfg.subspace.fisher_info.mode,
                    'forward_op_as_part_of_model': cfg.subspace.forward_op_as_part_of_model,
                    'update_curvature_ema': cfg.subspace.fisher_info.update_curvature_ema,
                    'curvature_ema_kwargs': OmegaConf.to_object(cfg.subspace.fisher_info.curvature_ema_kwargs),
                    'adaptive_damping_kwargs': OmegaConf.to_object(cfg.subspace.fisher_info.adaptive_damping_kwargs),
                })
            fisher_info.reset_fisher_matrix()

        subspace.init_parameters()
        recon = reconstructor.reconstruct(
            noisy_observation=observation,
            filtbackproj=filtbackproj,
            ground_truth=ground_truth,
            fisher_info=fisher_info,
            recon_from_randn=cfg.dip.recon_from_randn,
            log_path=cfg.dip.log_path,
            optim_kwargs=optim_kwargs
        )

        print('Subspace DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon[0, 0].cpu().numpy(), ground_truth[0, 0].cpu().numpy()))
        
        torch.save({
                    'reconstruction': recon.cpu(), 
                    'ground_truth': ground_truth.cpu(), 
                    'filtbackproj':  filtbackproj.cpu(), 
                    'observation': observation.cpu()
                }, f'recon_info_{i}.pt'
            )
        torch.save(subspace.state_dict(), f'subspace_state_dict_{i}.pt')

if __name__ == '__main__':
    coordinator()
