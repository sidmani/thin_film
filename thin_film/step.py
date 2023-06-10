import math
import numpy as np
from .kernel import grad_W_spiky, W_spline4
import thin_film.physics as physics
from itertools import cycle
from .util import chunk


def update_fields(r, u, Gamma, num_h, chunk, constants):
    divergence = np.empty((chunk[1] - chunk[0], 1))
    curvature = np.empty_like(divergence)
    new_Gamma = np.empty_like(divergence)

    for i in range(*chunk):
        rij = r - r[i]
        # compute the neighborhood
        r_len = np.sqrt(np.sum(rij**2, axis=1))
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        num_h_nb = num_h[nb]

        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        grad_kernel_reduced = 2 * np.sqrt(np.sum(grad_kernel**2, axis=1)) / r_len[nb]

        divergence[i - chunk[0]] = physics.compute_divergence(
            constants.V, num_h_nb, u[nb] - u[i], grad_kernel
        )
        curvature[i - chunk[0]] = physics.compute_curvature(
            constants.V, num_h_nb, num_h[i], grad_kernel_reduced
        )
        new_Gamma[i - chunk[0]] = physics.compute_surfactant_diffusion(
            constants.V,
            num_h_nb,
            Gamma[nb],
            Gamma[i],
            constants.alpha_c,
            constants.delta_t,
            grad_kernel_reduced,
        )

    return divergence, curvature, new_Gamma


def compute_forces(r, u, num_h, pressure, surface_tension, chunk, constants):
    force = np.zeros((chunk[1] - chunk[0], 2))
    new_num_h = np.empty((chunk[1] - chunk[0], 1))

    for i in range(*chunk):
        rij = r - r[i]
        # compute the neighborhood
        r_len = np.sqrt(np.sum(rij**2, axis=1))
        # create a neighborhood with and without the current particle
        inclusive_nb = r_len < constants.nb_threshold
        nb = inclusive_nb & (r_len > 0)

        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        grad_kernel_reduced = (
            2 * np.sqrt(np.sum(grad_kernel**2, axis=1)) / r_len[nb]
        )[:, None]

        # TODO: vorticity confinement force
        # pressure force
        force[i - chunk[0]] += physics.pressure_force(
            constants.V, num_h[nb], pressure[nb], num_h[i], pressure[i], grad_kernel
        )
        force[i - chunk[0]] += physics.marangoni_force(
            constants.V,
            num_h[nb],
            surface_tension[nb],
            num_h[i],
            surface_tension[i],
            grad_kernel,
        )
        # capillary force is normal to plane; ignored
        # viscosity force (the part in the plane)
        force[i - chunk[0]] += physics.viscosity_force(
            constants.V, constants.mu, u[nb] - u[i], num_h[nb], grad_kernel_reduced
        )

        # update numerical height
        # according to the paper this needs to be computed after the positions are updated
        # i.e. with a new neighborhood
        new_num_h[i - chunk[0]] = constants.V * np.sum(
            W_spline4(r_len[inclusive_nb], constants.nb_threshold)
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
    surface_tension = physics.compute_surface_tension(
        constants.gamma_0, constants.gamma_a, Gamma
    )

    chunk_size = math.ceil(r.shape[0] / len(pool._pool))
    result = pool.starmap(
        update_fields,
        map(
            lambda idx: [
                r,
                u,
                Gamma,
                num_h,
                (idx, min(idx + chunk_size, r.shape[0])),
                constants,
            ],
            range(0, r.shape[0], chunk_size),
        ),
    )
    transposed_result = list(map(list, zip(*result)))

    divergence = np.concatenate(transposed_result[0])
    curvature = np.concatenate(transposed_result[1])
    new_Gamma = np.concatenate(transposed_result[2])

    pressure = physics.compute_pressure(
        num_h,
        constants.h_0,
        constants.alpha_h,
        constants.alpha_k,
        constants.alpha_d,
        surface_tension,
        curvature,
        divergence,
    )
    adv_h += -adv_h * divergence * constants.delta_t

    result = pool.starmap(
        compute_forces,
        map(
            lambda idx: [
                r,
                u,
                num_h,
                pressure,
                surface_tension,
                (idx, min(idx + chunk_size, r.shape[0])),
                constants,
            ],
            range(0, r.shape[0], chunk_size),
        ),
    )
    transposed_result = list(map(list, zip(*result)))
    force = np.concatenate(transposed_result[0])
    new_num_h = np.concatenate(transposed_result[1])

    # TODO: updating by half should improve stability
    # 5. update velocity
    u += constants.delta_t / constants.m * force
    # 6. update position
    r += u * constants.delta_t

    return r, u, new_Gamma, new_num_h, adv_h
