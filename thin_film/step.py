import numpy as np
import scipy.constants
from sklearn.neighbors import KDTree
from .kernel import grad_W_spiky, W_spiky
from .fork_pdb import fork_pdb
from .util import chunk_starmap


def generate_nb(r, chunk, kdtree, nb_threshold, delete_center=True):
    nb_idxs, nb_dists = kdtree.query_radius(
        r[chunk[0] : chunk[1]], nb_threshold, return_distance=True
    )

    for i in range(*chunk):
        nb_idx = nb_idxs[i - chunk[0]]
        if delete_center:
            curr_idx = np.where(nb_idx == i)
            nb_idx = np.delete(nb_idx, curr_idx)
            nb_dist = np.delete(nb_dists[i - chunk[0]], curr_idx)
        else:
            nb_dist = nb_dists[i - chunk[0]]

        yield i, nb_idx, nb_dist


def get_numerical_height(chunk, query_pts, kdtree, constants):
    num_h = np.empty((chunk[1] - chunk[0],))
    for i, _, nb_dist in generate_nb(
        query_pts, chunk, kdtree, constants.nb_threshold, delete_center=False
    ):
        # evaluate the kernel on each neigborhood to compute the numerical height
        num_h[i - chunk[0]] = (
            constants.V * W_spiky(constants.nb_threshold, nb_dist).sum()
        )

    return (num_h,)


def update_fields(chunk, r, u, Gamma, num_h, kdtree, constants):
    divergence = np.empty((chunk[1] - chunk[0],))
    curvature = np.empty_like(divergence)
    new_Gamma = np.empty_like(divergence)
    vorticity = np.empty_like(divergence)

    for i, nb_idx, nb_dist in generate_nb(r, chunk, kdtree, constants.nb_threshold):
        rij = r[nb_idx] - r[i]
        uij = u[nb_idx] - u[i]
        num_h_nb = num_h[nb_idx]

        # compute the gradient of the smoothing kernel
        grad_kernel = grad_W_spiky(rij, constants.nb_threshold, nb_dist)
        grad_kernel_reduced = 2 * np.linalg.norm(grad_kernel, axis=1) / nb_dist

        vorticity[i - chunk[0]] = -constants.V * np.sum(
            np.cross(uij, grad_kernel) / num_h_nb
        )

        # uij is the velocity in the outwards radial direction
        # so the dot product will be negative if uij points outward
        # so the divergence needs an extra negative sign
        divergence[i - chunk[0]] = -constants.V * np.sum(
            np.sum((uij * grad_kernel), axis=1) / num_h_nb
        )

        curvature[i - chunk[0]] = constants.V * np.sum(
            (num_h_nb - num_h[i]) / num_h_nb * grad_kernel_reduced
        )

        # because of the product here, if Gamma is uniform to begin with it never changes
        new_Gamma[i - chunk[0]] = Gamma[i] + (
            constants.surfactant_diffusion_coefficient
            * constants.delta_t
            * np.sum(
                constants.V
                / num_h_nb
                * (Gamma[nb_idx] - Gamma[i])
                * grad_kernel_reduced
            )
        )

    return divergence, curvature, new_Gamma, vorticity


def compute_forces(
    chunk,
    r,
    u,
    num_h,
    vorticity,
    pressure,
    surface_tension,
    kdtree,
    constants,
):
    force = np.empty((chunk[1] - chunk[0], 2))

    for i, nb_idx, nb_dist in generate_nb(r, chunk, kdtree, constants.nb_threshold):
        rij = r[nb_idx] - r[i]
        uij = u[nb_idx] - u[i]
        num_h_nb = num_h[nb_idx]

        grad_kernel = grad_W_spiky(rij, constants.nb_threshold, nb_dist)
        grad_kernel_reduced = (2 * np.linalg.norm(grad_kernel, axis=1) / nb_dist)[
            :, np.newaxis
        ]

        viscosity_force = (
            constants.V**2
            * constants.viscosity
            * np.sum(uij / num_h_nb[:, np.newaxis] * grad_kernel_reduced, axis=0)
        )

        pressure_force = (
            2
            * constants.V**2
            * np.sum(
                num_h[i]
                * (pressure[i] / num_h[i] ** 2 + pressure[nb_idx] / num_h_nb**2)[
                    :, np.newaxis
                ]
                * grad_kernel,
                axis=0,
            )
        )

        marangoni_force = (
            constants.V
            / num_h[i]
            * np.sum(
                ((surface_tension[nb_idx] - surface_tension[i]) / num_h_nb)[
                    :, np.newaxis
                ]
                * grad_kernel
                * constants.V,
                axis=0,
            )
        )

        # extra forces
        vorticity_arg = constants.V * np.sum(
            (vorticity[nb_idx])[:, np.newaxis] * grad_kernel, axis=0
        )
        vorticity_norm = np.linalg.norm(vorticity_arg)
        if vorticity_norm == 0:
            vorticity_force = 0
        else:
            vorticity_lhs = vorticity_arg / vorticity_norm
            vorticity_force = (
                constants.vorticity
                * vorticity[i]
                * (vorticity_lhs[[1, 0]] * np.array([1, -1]))
            )

        gravity = np.array([0, -9.8 * constants.m]) * 1e-4

        force[i - chunk[0]] = (
            viscosity_force
            + pressure_force
            # + marangoni_force
            + vorticity_force
            + gravity
        )

    return (force,)


# enforce boundaries by reflecting particles
def boundary_reflect(r, u):
    exit_left = r[:, 0] < 0
    exit_right = r[:, 0] > 1
    exit_bottom = r[:, 1] < 0
    exit_top = r[:, 1] > 1

    r[:, 0] = np.where(exit_left, -r[:, 0], r[:, 0])
    r[:, 0] = np.where(exit_right, 2 - r[:, 0], r[:, 0])
    r[:, 1] = np.where(exit_bottom, -r[:, 1], r[:, 1])
    r[:, 1] = np.where(exit_top, 2 - r[:, 1], r[:, 1])

    u[:, 0] *= np.where(exit_left, -1, 1)
    u[:, 0] *= np.where(exit_right, -1, 1)
    u[:, 1] *= np.where(exit_bottom, 1, -1)
    u[:, 1] *= np.where(exit_top, 1, -1)


def step(r, u, Gamma, constants, pool, max_chunk_size=500):
    # generate a kd-tree to speed up nearest neighbor search
    kdtree = KDTree(r)

    # the surface tension is that of water (72e-3 N/m) minus the change caused by the surfactant
    surface_tension = 72e-3 - 293.15 * scipy.constants.R * Gamma

    (num_h,) = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=get_numerical_height,
        constant_args=[r, kdtree, constants],
        max_chunk_size=max_chunk_size,
    )

    divergence, curvature, new_Gamma, vorticity = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=update_fields,
        constant_args=[r, u, Gamma, num_h, kdtree, constants],
        max_chunk_size=max_chunk_size,
    )

    pressure = (
        # might make more sense to only have a pressure to spread out
        # instead of using a rest height, because the rest height is not physical
        # under gravitational influence
        constants.stiffness * (num_h / constants.rest_height)
        + constants.alpha_k * surface_tension * curvature
        - constants.alpha_d * divergence
    )

    (force,) = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=compute_forces,
        constant_args=[
            r,
            u,
            num_h,
            vorticity,
            pressure,
            surface_tension,
            kdtree,
            constants,
        ],
        max_chunk_size=max_chunk_size,
    )

    # update velocity and position
    u += constants.delta_t / constants.m * force
    r += u * constants.delta_t
    boundary_reflect(r, u)

    return r, u, new_Gamma
