"""
Based on https://github.com/educating-dip/bayes_dip/blob/main/bayes_dip/data/trafo/parallel_beam_2d_ray_trafo.py.
Provides :class:`ParallelBeam2DRayTrafo`, as well as getters
for its matrix representation and a :class:`MatmulRayTrafo` implementation.
"""
from itertools import product
from typing import Tuple, Optional
import numpy as np
import odl
import scipy
import os
from odl.contrib.torch import OperatorModule
from tqdm import tqdm
from subspace_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from subspace_dip.data.trafo.matmul_ray_trafo import MatmulRayTrafo


def get_odl_ray_trafo_fan_beam_2d(
        im_shape: Tuple[int, int],
        num_angles: int,
        src_radius: float, 
        det_radius: float,
        first_angle_zero: bool = True,
        impl: str = 'astra_cuda') -> odl.tomo.RayTransform:

    space = odl.uniform_discr(
            [-im_shape[0] / 2, -im_shape[1] / 2],
            [im_shape[0] / 2, im_shape[1] / 2],
            im_shape,
            dtype='float32')

    default_odl_geometry = odl.tomo.cone_beam_geometry(
            space, num_angles=num_angles, src_radius=src_radius, det_radius=det_radius)

    if first_angle_zero:
        default_first_angle = (
                default_odl_geometry.motion_grid.coord_vectors[0][0])
        angle_partition = odl.uniform_partition_fromgrid(
                odl.discr.grid.RectGrid(
                        default_odl_geometry.motion_grid.coord_vectors[0]
                        - default_first_angle))

        geometry = odl.tomo.FanBeamGeometry(
                apart=angle_partition,
                dpart=default_odl_geometry.det_partition,
                src_radius=src_radius, 
                det_radius=det_radius
                )
    else:
        geometry = default_odl_geometry

    odl_ray_trafo = odl.tomo.RayTransform(
                space, geometry, impl=impl)

    return odl_ray_trafo


class FanBeam2DRayTrafo(BaseRayTrafo):
    """
    Ray transform implemented via ODL.

    Adjoint computations use the back-projection (might be slightly inaccurate).
    """

    def __init__(self,
            im_shape: Tuple[int, int],
            num_angles: int,
            src_radius: float, 
            det_radius: float, 
            use_norm_op: bool = False, 
            first_angle_zero: bool = True,
            angular_sub_sampling: int = 1,
            impl: str = 'astra_cuda'):
        
        odl_ray_trafo_full = get_odl_ray_trafo_fan_beam_2d(
                im_shape, num_angles, 
                src_radius=src_radius, 
                det_radius=det_radius, 
                first_angle_zero=first_angle_zero,
                impl=impl)
        
        odl_ray_trafo_non_scaled = odl.tomo.RayTransform(
                odl_ray_trafo_full.domain,
                odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=impl)
        odl_fbp_non_scaled = odl.tomo.fbp_op(odl_ray_trafo_non_scaled, filter_type='Hann')

        odl_ray_trafo = odl_ray_trafo_non_scaled
        odl_fbp = odl_fbp_non_scaled
        if use_norm_op:
            self.norm_const = 1 / odl.power_method_opnorm(
                odl_ray_trafo_non_scaled
            )
            odl_ray_trafo = odl_ray_trafo_non_scaled * self.norm_const
            odl_fbp = odl_fbp_non_scaled / self.norm_const
        
        obs_shape = odl_ray_trafo_non_scaled.range.shape

        super().__init__(im_shape=im_shape, obs_shape=obs_shape)

        self.odl_ray_trafo = odl_ray_trafo
        self._angles = odl_ray_trafo_non_scaled.geometry.angles
        self.ray_trafo_module = OperatorModule(odl_ray_trafo)
        self.ray_trafo_module_adj = OperatorModule(odl_ray_trafo.adjoint)
        self.fbp_module = OperatorModule(odl_fbp)

    @property
    def angles(self) -> np.ndarray:
        """:class:`np.ndarray` : The angles (in radian)."""
        return self._angles

    def trafo(self, x):
        return self.ray_trafo_module(x)

    def trafo_adjoint(self, observation):
        return self.ray_trafo_module_adj(observation)

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint

    def fbp(self, observation):
        return self.fbp_module(observation)

def get_odl_ray_trafo_fan_beam_2d_matrix(
        im_shape: Tuple[int, int],
        num_angles: int,
        src_radius: float, 
        det_radius: float, 
        first_angle_zero: bool = True,
        angular_sub_sampling: int = 1,
        impl: str = 'astra_cuda',
        assemble_as_sparse: bool = True,
        flatten: bool = True
        ) -> np.ndarray:

    odl_ray_trafo_full = get_odl_ray_trafo_fan_beam_2d(
            im_shape, num_angles, 
            src_radius=src_radius, 
            det_radius=det_radius, 
            first_angle_zero=first_angle_zero,
            impl=impl
        )
    odl_ray_trafo = odl.tomo.RayTransform(
            odl_ray_trafo_full.domain,
            odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=impl)
    obs_shape = odl_ray_trafo.range.shape

    if not assemble_as_sparse: 
        matrix = np.zeros(obs_shape + im_shape, dtype=np.float32)
        x = np.zeros(im_shape, dtype=np.float32)
        for i0, i1 in tqdm(product(range(im_shape[0]), range(im_shape[1])),
                total=im_shape[0] * im_shape[1],
                desc='generating ray transform matrix'):
            x[i0, i1] = 1.
            matrix[:, :, i0, i1] = odl_ray_trafo_full(x)
            x[i0, i1] = 0.
    else:
        assert flatten
        assert angular_sub_sampling == 1
        matrix = scipy.sparse.dok_matrix(
            (np.prod(obs_shape), np.prod(im_shape)),
            dtype=np.float32)
        x = np.zeros(im_shape, dtype=np.float32)
        for i0, i1 in tqdm(product(range(im_shape[0]), range(im_shape[1])),
                total=im_shape[0] * im_shape[1],
                desc='generating ray transform matrix'):
            x[i0, i1] = 1.
            index = np.ravel_multi_index([i0,i1], (im_shape[0],im_shape[1]))
            op_x = odl_ray_trafo_full(x).asarray().flatten()
            non_zero_mask = op_x != 0. 
            matrix[non_zero_mask, index] = op_x[non_zero_mask]
            x[i0, i1] = 0.

    if assemble_as_sparse: 
        matrix = scipy.sparse.coo_matrix(matrix)
    
    # matrix = odl.operator.oputils.matrix_representation(
    #         odl_ray_trafo_full)

    if angular_sub_sampling != 1:
        matrix = matrix[::angular_sub_sampling]

    if flatten:
        matrix = matrix.reshape(-1, im_shape[0] * im_shape[1])

    return matrix


def get_fan_beam_2d_matmul_ray_trafo(
        im_shape: Tuple[int, int],
        num_angles: int,
        src_radius: float, 
        det_radius: float, 
        first_angle_zero: bool = True,
        angular_sub_sampling: int = 1,
        impl: str = 'astra_cuda',
        load_mat_from_path: Optional[str] = None,
        ) -> MatmulRayTrafo:
    """
    Return a :class:`bayes_dip.data.MatmulRayTrafo` with the matrix
    representation of an ODL 2D parallel beam ray transform.

    See documentation of :class:`ParallelBeam2DRayTrafo` for
    documentation of the parameters.
    """

    odl_ray_trafo_full = get_odl_ray_trafo_fan_beam_2d(
            im_shape, num_angles, src_radius=src_radius, det_radius=det_radius, first_angle_zero=first_angle_zero,
            impl=impl)
    odl_ray_trafo = odl.tomo.RayTransform(
            odl_ray_trafo_full.domain,
            odl_ray_trafo_full.geometry[::angular_sub_sampling], impl=impl)
    odl_fbp = odl.tomo.fbp_op(odl_ray_trafo, filter_type='Hann')

    obs_shape = odl_ray_trafo.range.shape
    angles = odl_ray_trafo.geometry.angles

    fbp_module = OperatorModule(odl_fbp)

    if load_mat_from_path is None: 
        matrix = get_odl_ray_trafo_fan_beam_2d_matrix(
                im_shape, num_angles, src_radius=src_radius, det_radius=det_radius, 
                first_angle_zero=first_angle_zero, 
                angular_sub_sampling=angular_sub_sampling, 
                impl=impl, 
                flatten=True
            )
        scipy.sparse.save_npz(f'fan_beam_{num_angles}_{src_radius}_{det_radius}_matrix.npz', matrix)
    else: 
        matrix = scipy.sparse.load_npz(os.path.join(load_mat_from_path, f'fan_beam_{num_angles}_{src_radius}_{det_radius}_matrix.npz'))

    ray_trafo = MatmulRayTrafo(im_shape, obs_shape, matrix, fbp_fun=fbp_module, angles=angles)

    return ray_trafo
