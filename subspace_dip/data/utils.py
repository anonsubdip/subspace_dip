from .trafo import (
    FanBeam2DRayTrafo,
    get_parallel_beam_2d_matmul_ray_trafo, get_fan_beam_2d_matmul_ray_trafo, get_walnut_2d_ray_trafo)

def get_ray_trafo(name, kwargs):

    if name == 'ellipses':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    elif name == 'rectangles':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    elif name == 'walnut_patches':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    elif name == 'cartoonset':
        ray_trafo = get_parallel_beam_2d_matmul_ray_trafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'])
    elif name in ('mayo', 'ellipses_mayo', 'lodopab_mayo_cropped', 'mayo_cropped'):
        # ray_trafo = get_fan_beam_2d_matmul_ray_trafo(
        #         im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
        #         angular_sub_sampling=kwargs['angular_sub_sampling'],
        #         src_radius=kwargs['src_radius'], 
        #         det_radius=kwargs['det_radius'], 
        #         load_mat_from_path=kwargs['load_mat_from_path']
        #         )
        ray_trafo = FanBeam2DRayTrafo(
                im_shape=kwargs['im_shape'], num_angles=kwargs['num_angles'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                src_radius=kwargs['src_radius'], 
                det_radius=kwargs['det_radius'], 
                use_norm_op=kwargs['use_norm_op']
                )
    elif name == 'walnut':
        ray_trafo = get_walnut_2d_ray_trafo(
                data_path=kwargs['data_path'],
                matrix_path=kwargs['matrix_path'],
                walnut_id=kwargs['walnut_id'],
                orbit_id=kwargs['orbit_id'],
                angular_sub_sampling=kwargs['angular_sub_sampling'],
                proj_col_sub_sampling=kwargs['proj_col_sub_sampling'])
    else:
        raise ValueError

    return ray_trafo
