import numpy as np
from .kernel import grad_W_spiky, W_spiky
from .fork_pdb import fork_pdb
from .util import chunk_starmap
import scipy.constants


# the augmentation causes weird patterns which makes me think it's not working
def init_numerical_height(chunk, r, u, constants):
    num_h = np.empty((chunk[1] - chunk[0],))
    for i in range(*chunk):
        _, _, aug_r_len, _ = compute_augmented_nb(
            r, u, i, [], constants.nb_threshold, constants.bounds
        )
        inclusive_aug_r_len = np.concatenate([aug_r_len, np.array([0])])

        num_h[i - chunk[0]] = constants.V * np.sum(
            W_spiky(constants.nb_threshold, inclusive_aug_r_len)
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
        constant_args=[r, u, constants],
    )[0]
    adv_h = num_h.copy()

    return r, u, Gamma, num_h, adv_h


def compute_augmented_nb(r, u, i, arrs_to_augment, nb_threshold, bounds):
    # compute the neighborhood
    r_len = np.linalg.norm(r - r[i], axis=1)
    nb = (r_len < nb_threshold) & (r_len > 0)
    r_nb = r[nb]

    # reflect the neighborhood over the point horizontally and vertically
    p = r[i]
    r_nb_horizontal = p[0] - (r_nb[:, 0] - p[0])
    r_nb_vertical = p[1] - (r_nb[:, 1] - p[1])

    # find the points whose reflections are over the boundaries
    augment_horizontal = (r_nb_horizontal < bounds[0]) | (r_nb_horizontal > bounds[2])
    # exclude augment_horizontal from augment_vertical to avoid double-counting
    augment_vertical = (
        (r_nb_vertical < bounds[1]) | (r_nb_vertical > bounds[3])
    ) & ~augment_horizontal

    # TODO: can short circuit if no augmentation needed
    result = []
    for a in arrs_to_augment:
        a_nb = a[nb]
        result.append(
            np.concatenate(
                [
                    a_nb,
                    a_nb[augment_horizontal],
                    a_nb[augment_vertical],
                ]
            )
        )

    aug_r = np.concatenate(
        [
            r_nb,
            np.column_stack([r_nb_horizontal, r_nb[:, 1]])[augment_horizontal],
            np.column_stack([r_nb[:, 0], r_nb_vertical])[augment_vertical],
        ]
    )

    u_nb = u[nb]
    aug_u = np.concatenate(
        [
            u_nb,
            u_nb[augment_horizontal] * np.array([[-1, 1]]),
            u_nb[augment_vertical] * np.array([[1, -1]]),
        ]
    )

    r_len_nb = r_len[nb]
    aug_r_len = np.concatenate(
        [
            r_len_nb,
            r_len_nb[augment_horizontal],
            r_len_nb[augment_vertical],
        ]
    )

    # the paper sets r to point radially inwards and u to point outwards
    # but we're going to set both pointing outwards because it makes more sense
    return aug_r - r[i], aug_u - u[i], aug_r_len, result


def update_fields(chunk, r, u, Gamma, num_h, constants):
    divergence = np.empty((chunk[1] - chunk[0],))
    curvature = np.empty_like(divergence)
    new_Gamma = np.empty_like(divergence)

    for i in range(*chunk):
        # construct the augmented neighborhood
        aug_rij, aug_uij, aug_r_len, [aug_num_h, aug_Gamma] = compute_augmented_nb(
            r, u, i, [num_h, Gamma], constants.nb_threshold, constants.bounds
        )

        # compute the gradient of the smoothing kernel
        # TODO: possible that this operator needs to take height curvature into account
        # the gradient returned by grad_kernel points radially inwards, toward r[i]
        grad_kernel = grad_W_spiky(aug_rij, constants.nb_threshold, aug_r_len)
        grad_kernel_reduced = 2 * np.linalg.norm(grad_kernel, axis=1) / aug_r_len

        # uij is the velocity in the outwards radial direction
        # so the dot product will be negative if uij points outward
        # so the divergence needs an extra negative sign
        divergence[i - chunk[0]] = -constants.V * np.sum(
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
        (
            aug_rij,
            aug_uij,
            aug_r_len,
            [aug_num_h, aug_pressure, aug_surface_tension],
        ) = compute_augmented_nb(
            r,
            u,
            i,
            [num_h, pressure, surface_tension],
            constants.nb_threshold,
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

        # update numerical height
        # according to the paper this needs to be computed after the positions are updated
        # i.e. with a new neighborhood
        # concat 0 because this neighborhood needs to include the current point
        new_num_h[i - chunk[0]] = constants.V * np.sum(
            W_spiky(constants.nb_threshold, np.concatenate([aug_r_len, np.array([0])]))
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
        - constants.alpha_d * divergence
    )

    # if divergence is positive, the height decreases
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
