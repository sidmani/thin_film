from collections import namedtuple
import torch
import math
from .kernel import W_spiky, grad_W_spiky, W_spline4
import pdb

def step(r, u, Gamma, num_h, adv_h, constants):
    # 1. compute preliminary variables
    surface_tension = constants.gamma_0 - constants.gamma_a * Gamma

    divergence = torch.zeros_like(Gamma)
    curvature = torch.zeros_like(Gamma)
    pressure = torch.zeros_like(Gamma)
    new_Gamma = torch.zeros_like(Gamma)
    for i in range(r.shape[0]):
        rij = r - r[i]
        uij = u - u[i]
        # compute the neighborhood
        r_len = torch.sqrt(torch.sum(rij**2, dim=1))
        nb = r_len < constants.nb_threshold

        # calculate divergence
        grad_kernel = grad_W_spiky(rij[nb], constants.kernel_h_spiky, r_len[nb])
        grad_kernel_reduced = (
            2 * torch.sqrt(torch.sum(grad_kernel**2, dim=1)) / r_len[nb]
        )
        divergence[i] = torch.sum(
            constants.V / num_h[nb] * torch.sum((uij[nb] * grad_kernel), dim=1)
        )

        # calculate local curvature
        curvature[i] = torch.sum(
            constants.V / num_h[nb] * (num_h[nb] - num_h[i]) * grad_kernel_reduced
        )

        # TODO: pressure
        pressure[i] = 0

        # update the advected height. could vectorize this
        # since it's not actually used for computation
        adv_h[i] += -adv_h[i] * divergence[i] * constants.delta_t

        # surfactant diffusion
        new_Gamma[i] += (
            constants.alpha_c
            * constants.delta_t
            * torch.sum(
                constants.V / num_h[nb] * (Gamma[nb] - Gamma[i]) * grad_kernel_reduced
            )
        )

    force = torch.zeros_like(r)
    new_num_h = torch.zeros_like(num_h)
    for i in range(r.shape[0]):
        rij = r - r[i]
        uij = u - u[i]
        # compute the neighborhood
        r_len = torch.sqrt(torch.sum(rij**2, dim=1))
        nb = r_len < constants.nb_threshold

        grad_kernel = grad_W_spiky(rij[nb], constants.kernel_h_spiky, r_len[nb])
        grad_kernel_reduced = (
            2 * torch.sqrt(torch.sum(grad_kernel**2, dim=1)) / r_len[nb]
        )
        # compute forces
        # pressure force
        # force[i] += (
        #     2
        #     * constants.V**2
        #     * torch.sum(
        #         num_h[i]
        #         * (pressure[i] / num_h[i] ** 2 + pressure[nb] / num_h[nb] ** 2)
        #         * grad_kernel,
        #         dim=0,
        #     )
        # )

        # Marangoni force
        force[i] += (
            constants.V
            / num_h[i]
            * torch.sum(
                constants.V
                / num_h[nb]
                * (surface_tension[nb] - surface_tension[i])
                * grad_kernel,
                dim=0,
            )
        )

        # capillary force is normal to plane; ignored
        # TODO: viscosity force (the part in the plane)

        # update numerical height
        new_num_h[i] = constants.V * torch.sum(
            W_spline4(r_len[nb], constants.kernel_h_spline4)
        )

    # 5. update velocity
    u += constants.delta_t / constants.m * force
    # 6. update position
    r += u * constants.delta_t

    return r, u, new_Gamma, new_num_h, adv_h


Constants = namedtuple(
    "Constants",
    [
        "V",
        "m",
        "nb_threshold",
        "gamma_0",
        "gamma_a",
        "delta_t",
        "kernel_h_spiky",
        "kernel_h_spline4",
        "alpha_c",
    ],
)


def run(particle_count, constants):
    with torch.no_grad():
        # position (r) and velocity (u)
        r = torch.zeros((particle_count, 2))
        u = torch.zeros((particle_count, 2))
        # surfactant concentration (Γ)
        Gamma = torch.zeros((particle_count))
        # numerical and advected height
        num_h = torch.zeros((particle_count))
        adv_h = torch.zeros((particle_count))

        for i in range(10):
            r, u, Gamma, num_h, adv_h = step(r, u, Gamma, num_h, adv_h, constants)


if __name__ == "__main__":
    run(
        10,
        Constants(
            V=1,
            m=1,
            nb_threshold=1,
            gamma_0=1,
            gamma_a=1,
            delta_t=1 / 60,
            kernel_h_spiky=1,
            kernel_h_spline4=1,
            alpha_c=1
        ),
    )
