import numpy as np
from .kernel import grad_W_spiky, W_spiky
from .fork_pdb import fork_pdb
from .util import chunk_starmap
import scipy.constants


def init_numerical_height(chunk, r, constants):
    num_h = np.empty((chunk[1] - chunk[0],))
    for i in range(*chunk):
        rij = r - r[i]
        r_len = np.sqrt(np.sum(rij**2, axis=1))
        inclusive_nb = r_len < constants.nb_threshold
        num_h[i - chunk[0]] = constants.V * np.sum(
            W_spiky(rij[inclusive_nb], constants.nb_threshold, r_len[inclusive_nb])
        )
    return (num_h,)


def init_values(constants, pool):
    bounds = constants.bounds
    r = np.random.rand(constants.particle_count, 2) * np.array(
        [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    ) + np.array([bounds[0], bounds[1]])
    u = np.zeros_like(r)

    # surfactant concentration (Γ)
    Gamma = (
        np.ones((constants.particle_count,))
        * constants.initial_surfactant_concentration
    )

    # numerical and advected height
    num_h = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=init_numerical_height,
        constant_args=[r, constants],
    )[0]
    adv_h = num_h.copy()

    return r, u, Gamma, num_h, adv_h


def update_fields(chunk, r, u, Gamma, num_h, constants):
    divergence = np.empty((chunk[1] - chunk[0],))
    curvature = np.empty_like(divergence)
    new_Gamma = np.empty_like(divergence)

    for i in range(*chunk):
        rij = r - r[i]
        # compute the neighborhood
        r_len = np.linalg.norm(rij, axis=1)
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        num_h_nb = num_h[nb]

        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        grad_kernel_reduced = 2 * np.linalg.norm(grad_kernel, axis=1) / r_len[nb]

        divergence[i - chunk[0]] = constants.V * np.sum(
            np.sum(((u[nb] - u[i]) * grad_kernel), axis=1) / num_h_nb
        )

        curvature[i - chunk[0]] = constants.V * np.sum(
            (num_h_nb - num_h[i]) / num_h_nb * grad_kernel_reduced
        )

        new_Gamma[i - chunk[0]] = Gamma[i] + (
            constants.surfactant_diffusion_coefficient
            * constants.delta_t
            * np.sum(
                constants.V / num_h_nb * (Gamma[nb] - Gamma[i]) * grad_kernel_reduced
            )
        )

    return divergence, curvature, new_Gamma


def compute_forces(chunk, r, u, num_h, pressure, surface_tension, constants):
    force = np.zeros((chunk[1] - chunk[0], 2))
    new_num_h = np.empty((chunk[1] - chunk[0],))

    for i in range(*chunk):
        rij = r - r[i]

        # compute the neighborhood
        r_len = np.linalg.norm(rij, axis=1)
        # create a neighborhood with and without the current particle
        inclusive_nb = r_len < constants.nb_threshold
        nb = inclusive_nb & (r_len > 0)

        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        grad_kernel_reduced = (
            2 * np.sqrt(np.sum(grad_kernel**2, axis=1)) / r_len[nb]
        )[:, None]

        # TODO: vorticity confinement force
        pressure_force = (
            2
            * constants.V**2
            * np.sum(
                num_h[i]
                * (pressure[i] / num_h[i] ** 2 + pressure[nb] / num_h[nb] ** 2)[
                    :, np.newaxis
                ]
                * grad_kernel,
                axis=0,
            )
        )

        marangoni_force = (
            constants.V**2
            / num_h[i]
            * np.sum(
                ((surface_tension[nb] - surface_tension[i]) / num_h[nb])[:, np.newaxis]
                * grad_kernel,
                axis=0,
            )
        )

        viscosity_force = (
            constants.V**2
            * constants.mu
            * np.sum((u[nb] - u[i]) / (num_h[nb])[:, np.newaxis] * grad_kernel_reduced)
        )
        force[i - chunk[0]] = pressure_force + marangoni_force + viscosity_force

        # update numerical height
        # according to the paper this needs to be computed after the positions are updated
        # i.e. with a new neighborhood
        new_num_h[i - chunk[0]] = constants.V * np.sum(
            # W_spline4(r_len[inclusive_nb], constants.nb_threshold)
            W_spiky(rij[inclusive_nb], constants.nb_threshold, r_len[inclusive_nb])
        )

    return force, new_num_h


def step(
    r,
    u,
    Gamma,
    num_h,
    adv_h,
    constants,
    pool,
):
    surface_tension = (
        constants.pure_surface_tension - 293.15 * scipy.constants.R * Gamma
    )

    divergence, curvature, new_Gamma = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=update_fields,
        constant_args=[r, u, Gamma, num_h, constants],
    )

    pressure = (
        constants.alpha_h * (num_h / constants.h_0 - 1)
        + constants.alpha_k * surface_tension * curvature
        + constants.alpha_d * divergence
    )

    # TODO: seems like there could be a sign issue here
    adv_h += -adv_h * divergence * constants.delta_t

    force, new_num_h = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=compute_forces,
        constant_args=[r, u, num_h, pressure, surface_tension, constants],
    )

    # TODO: updating by half should improve stability
    # 5. update velocity
    u += constants.delta_t / constants.m * force
    # 6. update position
    r += u * constants.delta_t

    return r, u, new_Gamma, new_num_h, adv_h
