from collections import namedtuple
import torch
from .kernel import grad_W_spiky, W_spline4
from .render import render_frame
from tqdm import tqdm

def step(r, u, Gamma, num_h, adv_h, constants):
    divergence = torch.zeros_like(Gamma)
    curvature = torch.zeros_like(Gamma)
    pressure = torch.zeros_like(Gamma)
    new_Gamma = torch.zeros_like(Gamma)
    surface_tension = constants.gamma_0 - constants.gamma_a * Gamma

    for i in tqdm(range(r.shape[0]), position=1, leave=False):
        rij = r - r[i]
        uij = u - u[i]
        # compute the neighborhood
        r_len = torch.sqrt(torch.sum(rij**2, dim=1))
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        # divergence
        grad_kernel = grad_W_spiky(rij[nb], constants.kernel_h_spiky, r_len[nb])
        grad_kernel_reduced = (
            2 * torch.sqrt(torch.sum(grad_kernel**2, dim=1)) / r_len[nb]
        )
        divergence[i] = torch.sum(
            constants.V / num_h[nb] * torch.sum((uij[nb] * grad_kernel), dim=1)
        )

        # local curvature
        curvature[i] = torch.sum(
            constants.V / num_h[nb] * (num_h[nb] - num_h[i]) * grad_kernel_reduced
        )

        # pressure
        pressure[i] = (
            constants.alpha_h * (num_h[i] / constants.h_0 - 1)
            + constants.alpha_k * surface_tension[i] * curvature[i]
            + constants.alpha_d * divergence[i]
        )

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
    new_num_h = torch.ones_like(num_h)
    for i in tqdm(range(r.shape[0]), position=1, leave=False):
        rij = r - r[i]
        uij = u - u[i]
        # compute the neighborhood
        r_len = torch.sqrt(torch.sum(rij**2, dim=1))
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        grad_kernel = grad_W_spiky(rij[nb], constants.kernel_h_spiky, r_len[nb])
        grad_kernel_reduced = (
            2 * torch.sqrt(torch.sum(grad_kernel**2, dim=1)) / r_len[nb]
        )
        # compute forces
        # pressure force
        force[i] += (
            2
            * constants.V**2
            * torch.sum(
                num_h[i]
                * (pressure[i] / num_h[i] ** 2 + pressure[nb] / num_h[nb] ** 2)[:, None]
                * grad_kernel,
                dim=0,
            )
        )

        # Marangoni force
        # force[i] += (
        #     constants.V
        #     / num_h[i]
        #     * torch.sum(
        #         constants.V
        #         / num_h[nb]
        #         * (surface_tension[nb] - surface_tension[i])
        #         * grad_kernel,
        #         dim=0,
        #     )
        # )

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
        "alpha_c", # surfactant diffusion coefficient
        "alpha_h",
        "alpha_k",
        "alpha_d",
        "h_0",
    ],
)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def run(particle_count, steps, constants):
    bounds = (0, 0, 1, 1)
    frames = []

    with torch.no_grad():
        # position (r) and velocity (u)
        r = torch.rand((particle_count, 2)) * torch.tensor(
            [bounds[2] - bounds[0], bounds[3] - bounds[1]]
        ) + torch.tensor([bounds[0], bounds[1]])
        u = torch.zeros((particle_count, 2))
        # surfactant concentration (Γ)
        Gamma = torch.zeros((particle_count))
        # numerical and advected height
        num_h = torch.ones((particle_count))
        adv_h = torch.ones((particle_count))

        for i in tqdm(range(steps), position=0, leave=False):
            r, u, Gamma, num_h, adv_h = step(r, u, Gamma, num_h, adv_h, constants)
            frames.append(render_frame(r, adv_h, (1024, 1024), bounds))

    im1 = plt.imshow(frames[0])

    def update(i):
        im1.set_data(frames[i])

    ani = FuncAnimation(plt.gcf(), update, interval=200)
    plt.show()


if __name__ == "__main__":
    run(
        10000,
        10,
        Constants(
            V=1,
            m=1,
            nb_threshold=0.2,
            gamma_0=1,
            gamma_a=1,
            delta_t=1 / 60,
            kernel_h_spiky=3,
            kernel_h_spline4=3,
            alpha_c=1,
            alpha_d=1,
            alpha_h=1,
            alpha_k=1,
            h_0=1,
        ),
    )
