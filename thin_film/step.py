import numpy as np
import scipy.constants
from sklearn.neighbors import KDTree
from .kernel import grad_W_spiky, W_spiky
from .fork_pdb import fork_pdb
from .util import chunk_starmap
import scipy.constants


def get_numerical_height(chunk, r, kdtree, constants, inner_chunk_size=500):
    num_h = np.empty((chunk[1] - chunk[0],))

    for i in range(chunk[0], chunk[1], inner_chunk_size):
        upper_bound = min(i + inner_chunk_size, chunk[1])
        _, nb_dist = kdtree.query_radius(
            r[i:upper_bound],
            constants.nb_threshold,
            return_distance=True,
        )

        num_h[i - chunk[0] : upper_bound - chunk[0]] = constants.V * np.array(
            [W_spiky(constants.kernel_h, nb_dist_j).sum() for nb_dist_j in nb_dist]
        )

    return (num_h,)


def update_fields(chunk, r, u, Gamma, num_h, kdtree, constants):
    divergence = np.empty((chunk[1] - chunk[0],))
    curvature = np.empty_like(divergence)
    new_Gamma = np.empty_like(divergence)

    for i in range(*chunk):
        nb_idx = kdtree.query_radius(r[i].reshape(1, -1), constants.nb_threshold)[0]
        # delete the current point from the neighborhood
        nb_idx = np.delete(nb_idx, np.where(nb_idx == i))

        rij = r[nb_idx] - r[i]
        r_len = np.linalg.norm(rij, axis=1)
        uij = u[nb_idx] - u[i]
        num_h_nb = num_h[nb_idx]

        if r_len.min() == 0:
            raise Exception("Multiple particles found at the same point!")

        # compute the gradient of the smoothing kernel
        # TODO: possible that this operator needs to take height curvature into account
        # the gradient returned by grad_kernel points radially inwards, toward r[i]
        grad_kernel = grad_W_spiky(rij, constants.kernel_h, r_len)
        grad_kernel_reduced = 2 * np.linalg.norm(grad_kernel, axis=1) / r_len

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

    return divergence, curvature, new_Gamma


def compute_forces(chunk, r, u, num_h, pressure, surface_tension, kdtree, constants):
    force = np.zeros((chunk[1] - chunk[0], 2))

    for i in range(*chunk):
        nb_idx = kdtree.query_radius(r[i].reshape(1, -1), constants.nb_threshold)[0]
        nb_idx = np.delete(nb_idx, np.where(nb_idx == i))

        rij = r[nb_idx] - r[i]
        r_len = np.linalg.norm(rij, axis=1)
        uij = u[nb_idx] - u[i]
        num_h_nb = num_h[nb_idx]

        if r_len.min() == 0:
            raise Exception("Multiple particles found at the same point!")

        grad_kernel = grad_W_spiky(rij, constants.kernel_h, r_len)
        grad_kernel_reduced = (2 * np.sqrt(np.sum(grad_kernel**2, axis=1)) / r_len)[
            :, None
        ]

        viscosity_force = (
            constants.V**2
            * constants.mu
            * np.sum(uij / num_h_nb[:, np.newaxis] * grad_kernel_reduced, axis=0)
        )

        # TODO: vorticity confinement force and capillary force
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

        # marangoni_force = (
        #     constants.V**2
        #     / num_h[i]
        #     * np.sum(
        #         ((aug_surface_tension - surface_tension[i]) / aug_num_h)[:, np.newaxis]
        #         * grad_kernel,
        #         axis=0,
        #     )
        # )

        # viscosity_force = (
        #     constants.V**2
        #     * constants.mu
        #     * np.sum((aug_uij) / (aug_num_h)[:, np.newaxis] * grad_kernel_reduced, axis=0)
        # )

        # force[i - chunk[0]] = pressure_force + marangoni_force + viscosity_force
        force[i - chunk[0]] = pressure_force

        force[i - chunk[0]] = viscosity_force + pressure_force

    return (force,)


# enforce boundaries by reflecting particles outside the bounds
def boundary_reflect(r, u, bounds):
    exit_left = r[:, 0] < bounds[0]
    exit_right = r[:, 0] > bounds[2]
    exit_bottom = r[:, 1] < bounds[1]
    exit_top = r[:, 1] > bounds[3]

    r[:, 0] = np.where(exit_left, bounds[0] - r[:, 0], r[:, 0])
    r[:, 0] = np.where(exit_right, 2 * bounds[2] - r[:, 0], r[:, 0])
    r[:, 1] = np.where(exit_bottom, bounds[1] - r[:, 1], r[:, 1])
    r[:, 1] = np.where(exit_top, 2 * bounds[3] - r[:, 1], r[:, 1])

    u[:, 0] *= np.where(exit_left, -1, 1)
    u[:, 0] *= np.where(exit_right, -1, 1)
    u[:, 1] *= np.where(exit_bottom, 1, -1)
    u[:, 1] *= np.where(exit_top, 1, -1)


def step(
    r,
    u,
    Gamma,
    adv_h,
    constants,
    pool,
):
    # generate a kd-tree to speed up nearest neighbor search
    kdtree = KDTree(r)

    # the surface tension is that of water (72e-3 N/m) minus the change caused by the surfactant
    surface_tension = 72e-3 - 293.15 * scipy.constants.R * Gamma

    (num_h,) = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=get_numerical_height,
        constant_args=[r, kdtree, constants],
    )

    if adv_h is None:
        adv_h = num_h.copy()

    divergence, curvature, new_Gamma = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=update_fields,
        constant_args=[r, u, Gamma, num_h, kdtree, constants],
    )

    pressure = (
        constants.stiffness * (num_h / constants.rest_height - 1)
        # why is it multiplied by the curvature?
        + constants.alpha_k * surface_tension * curvature
        # is this sign correct?
        - constants.alpha_d * divergence
    )

    # if divergence is positive, the height decreases
    adv_h += -adv_h * divergence * constants.delta_t

    (force,) = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=compute_forces,
        constant_args=[r, u, num_h, pressure, surface_tension, kdtree, constants],
    )

    # TODO: updating by half should improve stability
    # update velocity and position
    u += constants.delta_t / constants.m * force
    r += u * constants.delta_t

    boundary_reflect(r, u, constants.bounds)

    return r, u, new_Gamma, adv_h
