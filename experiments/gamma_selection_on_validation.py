from omegaconf import DictConfig, OmegaConf

import os
import json
import copy
import difflib
import hydra
import numpy as np

from subspace_dip.utils import find_log_files, extract_tensorboard_scalars, print_dct, sorted_dict

def collect_runs_paths_per_gamma(base_paths, raise_on_cfg_diff=False):
    paths = {}
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    ref_cfg = None
    ignore_keys_in_cfg_diff = [
            'dip.optim.gamma', 'dip.torch_manual_seed']
    for base_path in base_paths:
        path = os.path.join(os.getcwd().partition('src')[0], base_path)
        for dirpath, dirnames, filenames in os.walk(path):
            if '.hydra' in dirnames:
                cfg = OmegaConf.load(
                        os.path.join(dirpath, '.hydra', 'config.yaml'))
                paths.setdefault(cfg.dip.optim.gamma, []).append(dirpath)

                if ref_cfg is None:
                    ref_cfg = copy.deepcopy(cfg)
                    for k in ignore_keys_in_cfg_diff:
                        OmegaConf.update(ref_cfg, k, None)
                    ref_cfg_yaml = OmegaConf.to_yaml(sorted_dict(OmegaConf.to_object(ref_cfg)))
                    ref_dirpath = dirpath
                else:
                    cur_cfg = copy.deepcopy(cfg)
                    for k in ignore_keys_in_cfg_diff:
                        OmegaConf.update(cur_cfg, k, None)
                    cur_cfg_yaml = OmegaConf.to_yaml(sorted_dict(OmegaConf.to_object(cur_cfg)))
                    try:
                        assert cur_cfg_yaml == ref_cfg_yaml
                    except AssertionError:
                        print('Diff between config at path {} and config at path {}'.format(ref_dirpath, dirpath))
                        differ = difflib.Differ()
                        diff = differ.compare(ref_cfg_yaml.splitlines(),
                                              cur_cfg_yaml.splitlines())
                        print('\n'.join(diff))
                        # print('\n'.join([d for d in diff if d.startswith('-') or d.startswith('+')]))
                        if raise_on_cfg_diff:
                            raise

    paths = {k:sorted(v) for k, v in sorted(paths.items()) if v}
    return paths

@hydra.main(config_path='hydra_cfg', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if not cfg.val.select_gamma_multirun_base_paths:
        raise ValueError

    runs = collect_runs_paths_per_gamma(
            cfg.val.select_gamma_multirun_base_paths)  # , raise_on_cfg_diff=False)  # -> check diff output manually
    print_dct(runs) # visualise runs and models checkpoints

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_run_paths_filename)), exist_ok=True)
    with open(cfg.val.run_paths_filename, 'w') as f:
        json.dump(runs, f, indent=1)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_results_filename)), exist_ok=True)

    infos = {}
    for i_run, (gamma, histories_path) in enumerate(runs.items()):
        psnr_histories = []
        samples_log_files = find_log_files(histories_path[0])
        for tb_path in samples_log_files:
            extracted_min_loss_psnr = extract_tensorboard_scalars(tb_path)['min_loss_output_psnr_scalars']
            psnr_histories.append(extracted_min_loss_psnr)
        
        median_psnr_output = np.median(psnr_histories, axis=0)
        psnr_steady = np.max(median_psnr_output[
                cfg.val.psnr_steady_start:cfg.val.psnr_steady_stop])
        infos[gamma] = {
                'PSNR_steady': psnr_steady, 'PSNR_0': median_psnr_output[0]}

        with open(cfg.val.select_gamma_results_filename, 'w') as f:
            json.dump(infos, f, indent=1)

    def key(info):
        return -info['PSNR_steady']

    infos = {k: v for k, v in sorted(infos.items(), key=lambda item: key(item[1]))}
    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_results_sorted_filename)), exist_ok=True)
    with open(cfg.val.select_gamma_results_sorted_filename, 'w') as f:
        json.dump(infos, f, indent=1)

if __name__ == '__main__':
    coordinator()