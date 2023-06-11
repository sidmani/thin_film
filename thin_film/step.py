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


def augment_nb(r, u, i, arrs_to_augment, nb, bounds):
    # x0, y0, x1, y1 = bounds
    r_nb = r[nb]
    # reflect the neighborhood over the point horizontally and vertically
    p = r[i]
    r_nb_horizontal = p[0] - (r_nb[:, 0] - p[0])
    r_nb_vertical = p[1] - (r_nb[:, 1] - p[1])

    # find the points whose reflections are over the boundaries
    augment_horizontal = (r_nb_horizontal < bounds[0]) | (r_nb_horizontal > bounds[2])
    augment_vertical = (r_nb_vertical < bounds[1]) | (r_nb_vertical > bounds[3])

    augment = augment_horizontal | augment_vertical
    result = []
    for a in arrs_to_augment:
        a_nb = a[nb]
        result.append(np.concatenate([a_nb, a_nb[augment]]))

    r_nb = r[nb]
    aug_r = np.concatenate(
        [
            r_nb,
            np.column_stack([r_nb_horizontal, r_nb[:, 1]])[augment_horizontal],
            np.column_stack([r_nb[:, 0], r_nb_vertical])[
                augment_vertical & ~augment_horizontal
            ],
        ]
    )

    u_nb = u[nb]
    aug_u = np.concatenate(
        [
            u_nb,
            u_nb[augment_horizontal] * np.array([[-1, 0]]),
            u_nb[augment_vertical & ~augment_horizontal] * np.array([[0, -1]]),
        ]
    )

    return aug_r - r[i], aug_u, result


def update_fields(chunk, r, u, Gamma, num_h, constants):
    divergence = np.empty((chunk[1] - chunk[0],))
    curvature = np.empty_like(divergence)
    new_Gamma = np.empty_like(divergence)

    for i in range(*chunk):
        # compute the neighborhood
        rij = r - r[i]
        r_len = np.linalg.norm(rij, axis=1)
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        # augment the neighborhood at the boundaries
        aug_rij, aug_uij, [aug_r_len, aug_num_h, aug_Gamma] = augment_nb(
            r, u - u[i], i, [r_len, num_h, Gamma], nb, constants.bounds
        )

        # compute the fields
        grad_kernel = grad_W_spiky(aug_rij, constants.nb_threshold, aug_r_len)
        grad_kernel_reduced = 2 * np.linalg.norm(grad_kernel, axis=1) / aug_r_len

        divergence[i - chunk[0]] = constants.V * np.sum(
            np.sum((aug_uij * grad_kernel), axis=1) / aug_num_h
        )

        curvature[i - chunk[0]] = constants.V * np.sum(
            (aug_num_h - num_h[i]) / aug_num_h * grad_kernel_reduced
        )

        new_Gamma[i - chunk[0]] = Gamma[i] + (
            constants.surfactant_diffusion_coefficient
            * constants.delta_t
            * np.sum(
                constants.V / aug_num_h * (aug_Gamma - Gamma[i]) * grad_kernel_reduced
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

        # augment the neighborhood at the boundaries
        (
            aug_rij,
            aug_uij,
            [aug_r_len, aug_num_h, aug_pressure, aug_surface_tension],
        ) = augment_nb(
            r,
            u - u[i],
            i,
            [r_len, num_h, pressure, surface_tension],
            nb,
            constants.bounds,
        )

        grad_kernel = grad_W_spiky(aug_rij, constants.nb_threshold, aug_r_len)
        grad_kernel_reduced = (
            2 * np.sqrt(np.sum(grad_kernel**2, axis=1)) / aug_r_len
        )[:, None]

        # TODO: vorticity confinement force
        pressure_force = (
            2
            * constants.V**2
            * np.sum(
                num_h[i]
                * (pressure[i] / num_h[i] ** 2 + aug_pressure / aug_num_h**2)[
                    :, np.newaxis
                ]
                * grad_kernel,
                axis=0,
            )
        )

        # try:
        #     if aug_rij.shape[0] > rij[nb].shape[0]:
        #         fork_pdb.set_trace()
        #     tmp_grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        #     unaug_pf = (
        #         2
        #         * constants.V**2
        #         * np.sum(
        #             num_h[i]
        #             * (pressure[i] / num_h[i] ** 2 + pressure[nb] / num_h[nb] ** 2)[
        #                 :, np.newaxis
        #             ]
        #             * tmp_grad_kernel,
        #             axis=0,
        #         )
        #     )
        # except Exception as e:
        #     fork_pdb.set_trace()

        marangoni_force = (
            constants.V**2
            / num_h[i]
            * np.sum(
                ((aug_surface_tension - surface_tension[i]) / aug_num_h)[:, np.newaxis]
                * grad_kernel,
                axis=0,
            )
        )

        viscosity_force = (
            constants.V**2
            * constants.mu
            * np.sum((aug_uij) / (aug_num_h)[:, np.newaxis] * grad_kernel_reduced)
        )
        force[i - chunk[0]] = pressure_force

        # update numerical height
        # according to the paper this needs to be computed after the positions are updated
        # i.e. with a new neighborhood
        inclusive_aug_rij = np.concatenate([aug_rij, np.array([[0, 0]])])
        inclusive_aug_r_len = np.concatenate([aug_r_len, np.array([0])])
        new_num_h[i - chunk[0]] = constants.V * np.sum(
            # W_spline4(r_len[inclusive_nb], constants.nb_threshold)
            W_spiky(inclusive_aug_rij, constants.nb_threshold, inclusive_aug_r_len)
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
    # aug_r, aug_u, [aug_Gamma, aug_num_h] = augment_particles(
    #     r, u, [Gamma, num_h], constants.bounds, constants.nb_threshold
    # )
    # fork_pdb.set_trace()

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
