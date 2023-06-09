import numpy as np
from .kernel import grad_W_spiky, W_spline4
import thin_film.physics as physics
from itertools import cycle
from .util import chunk


def update_fields(r, u, Gamma, num_h, constants):
    divergence = np.empty_like(Gamma)
    curvature = np.empty_like(Gamma)
    new_Gamma = np.empty_like(Gamma)

    for i in range(r.shape[0]):
        rij = r - r[i]
        # compute the neighborhood
        r_len = np.sqrt(np.sum(rij**2, axis=1))
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        num_h_nb = num_h[nb]

        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        grad_kernel_reduced = 2 * np.sqrt(np.sum(grad_kernel**2, axis=1)) / r_len[nb]

        divergence[i] = physics.compute_divergence(
            constants.V, num_h_nb, u[nb] - u[i], grad_kernel
        )
        curvature[i] = physics.compute_curvature(
            constants.V, num_h_nb, num_h[i], grad_kernel_reduced
        )
        new_Gamma[i] = physics.compute_surfactant_diffusion(
            constants.V,
            num_h_nb,
            Gamma[nb],
            Gamma[i],
            constants.alpha_c,
            constants.delta_t,
            grad_kernel_reduced,
        )

    return divergence, curvature, new_Gamma


def compute_forces(r, u, num_h, pressure, surface_tension, constants):
    force = np.zeros_like(r)
    new_num_h = np.empty_like(num_h)
    # compute forces
    for i in range(r.shape[0]):
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
        force[i] += physics.pressure_force(
            constants.V, num_h[nb], pressure[nb], num_h[i], pressure[i], grad_kernel
        )
        force[i] += physics.marangoni_force(
            constants.V,
            num_h[nb],
            surface_tension[nb],
            num_h[i],
            surface_tension[i],
            grad_kernel,
        )
        # capillary force is normal to plane; ignored
        # viscosity force (the part in the plane)
        force[i] += physics.viscosity_force(
            constants.V, constants.mu, u[nb] - u[i], num_h[nb], grad_kernel_reduced
        )

        # update numerical height
        new_num_h[i] = constants.V * np.sum(
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

    # TODO: if chunks are manageable the for loops can be removed
    chunks = chunk(500, [r, u, Gamma, num_h])
    result = pool.starmap(
        update_fields,
        zip(*chunks, cycle([constants])),
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

    chunks = chunk(500, [r, u, num_h, pressure, surface_tension])
    result = pool.starmap(compute_forces, zip(*chunks, cycle([constants])))
    transposed_result = list(map(list, zip(*result)))
    force = np.concatenate(transposed_result[0])
    new_num_h = np.concatenate(transposed_result[1])

    # TODO: updating by half should improve stability
    # 5. update velocity
    u += constants.delta_t / constants.m * force
    # 6. update position
    r += u * constants.delta_t

    return r, u, new_Gamma, new_num_h, adv_h
