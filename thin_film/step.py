import math
import numpy as np
from .kernel import grad_W_spiky, W_spline4, W_spiky
import thin_film.physics as physics
from .fork_pdb import fork_pdb


# send func chunks of data and reassemble the results into numpy arrays
def chunk_starmap(total_count, pool, func, constant_args, chunk_size=None):
    if chunk_size is None:
        chunk_size = math.ceil(total_count / len(pool._pool))

    result = pool.starmap(
        func,
        map(
            lambda idx: [(idx, min(idx + chunk_size, total_count)), *constant_args],
            range(0, total_count, chunk_size),
        ),
    )
    return tuple(map(np.concatenate, zip(*result)))


def init_numerical_height(chunk, r, constants):
    num_h = np.empty((chunk[1] - chunk[0],))
    for i in range(*chunk):
        rij = r - r[i]
        r_len = np.sqrt(np.sum(rij**2, axis=1))
        inclusive_nb = r_len < constants.nb_threshold
        # nb = (r_len < constants.nb_threshold) & (r_len > 0)
        num_h[i - chunk[0]] = constants.V * np.sum(
            # W_spline4(r_len[inclusive_nb], constants.nb_threshold)
            W_spiky(rij[inclusive_nb], constants.nb_threshold, r_len[inclusive_nb])
        )
    return (num_h,)


def init_values(constants, bounds, pool):
    r = np.random.rand(constants.particle_count, 2) * np.array(
        [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    ) + np.array([bounds[0], bounds[1]])
    u = np.zeros_like(r)

    # surfactant concentration (Γ)
    # TODO: this should probably not be zero
    Gamma = np.zeros((constants.particle_count,))

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


def compute_forces(chunk, r, u, num_h, pressure, surface_tension, constants):
    force = np.zeros((chunk[1] - chunk[0], 2))
    new_num_h = np.empty((chunk[1] - chunk[0],))

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
    surface_tension = physics.compute_surface_tension(
        constants.gamma_0, constants.gamma_a, Gamma
    )

    divergence, curvature, new_Gamma = chunk_starmap(
        total_count=constants.particle_count,
        pool=pool,
        func=update_fields,
        constant_args=[r, u, Gamma, num_h, constants],
    )

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
