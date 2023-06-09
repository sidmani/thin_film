from collections import namedtuple
import pdb
import scipy.constants
import torch
from .kernel import grad_W_spiky, W_spline4
from .render import render_frame
from tqdm import tqdm

def bad(t):
    return torch.any(torch.isnan(t) | torch.isinf(t)).item()

def step(r, u, Gamma, num_h, adv_h, constants):
    divergence = torch.empty_like(Gamma)
    curvature = torch.empty_like(Gamma)
    pressure = torch.empty_like(Gamma)
    new_Gamma = Gamma.clone()

    # gamma_a = ideal gas constant * room temperature
    surface_tension = constants.gamma_0 - (293.15 * scipy.constants.R) * Gamma

    for i in tqdm(range(r.shape[0]), position=1, leave=False):
        rij = r - r[i]
        uij = u - u[i]
        # compute the neighborhood
        r_len = torch.sqrt(torch.sum(rij**2, dim=1))
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        # divergence
        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
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

        # surfactant diffusion
        new_Gamma[i] += (
            constants.alpha_c
            * constants.delta_t
            * torch.sum(
                constants.V / num_h[nb] * (Gamma[nb] - Gamma[i]) * grad_kernel_reduced
            )
        )

    # pressure
    pressure = (
        constants.alpha_h * (num_h / constants.h_0 - 1)
        + constants.alpha_k * surface_tension * curvature
        + constants.alpha_d * divergence
    )

    # advected height
    adv_h += -adv_h * divergence * constants.delta_t

    force = torch.zeros_like(r)
    new_num_h = torch.empty_like(num_h)

    # compute forces
    for i in tqdm(range(r.shape[0]), position=1, leave=False):
        rij = r - r[i]
        uij = u - u[i]
        # compute the neighborhood
        r_len = torch.sqrt(torch.sum(rij**2, dim=1))
        # create a neighborhood with and without the current particle
        inclusive_nb = r_len < constants.nb_threshold
        nb = inclusive_nb & (r_len > 0)

        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        grad_kernel_reduced = (
            2 * torch.sqrt(torch.sum(grad_kernel**2, dim=1)) / r_len[nb]
        )[:, None]

        # TODO: vorticity confinement force
        # pressure force
        force[i] += (
            2
            * constants.V**2
            * torch.sum(
                num_h[i]
                * (pressure[i] / num_h[i] ** 2 + pressure[nb] / num_h[nb] ** 2)
                * grad_kernel,
                dim=0,
            )
        )
        if (bad(force)):
            pdb.set_trace()

        # Marangoni force
        force[i] += (
            constants.V**2
            / num_h[i]
            * torch.sum(
                (surface_tension[nb] - surface_tension[i]) / num_h[nb] * grad_kernel,
                dim=0,
            )
        )
        if (bad(force)):
            pdb.set_trace()
        # capillary force is normal to plane; ignored
        # viscosity force (the part in the plane)
        force[i] += (
            constants.V**2
            * constants.mu
            * torch.sum(uij[nb] / num_h[nb] * grad_kernel_reduced)
        )
        if (bad(force)):
            pdb.set_trace()
        # update numerical height
        new_num_h[i] = constants.V * torch.sum(
            W_spline4(r_len[inclusive_nb], constants.nb_threshold)
        )

    # TODO: updating by half should improve stability
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
        "delta_t",
        "alpha_c",  # surfactant diffusion coefficient
        "alpha_h",
        "alpha_k",
        "alpha_d",
        "h_0",
        "mu",
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
        Gamma = torch.zeros((particle_count, 1))
        # numerical and advected height
        num_h = torch.ones((particle_count, 1)) * constants.h_0
        adv_h = torch.ones((particle_count, 1)) * constants.h_0

        # TODO: num_h and adv_h are off by a few orders of magnitude
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
        1000,
        20,
        Constants(
            V=1e-7,
            m=1e-7,
            nb_threshold=0.01,
            gamma_0=72e-3,  # surface tension of water at room temp = 72 mN/m
            delta_t=1 / 60,  # 60 fps
            alpha_c=1e-8, # diffusion coefficient of surfactant
            alpha_d=1e-8,
            alpha_h=1e-8,
            alpha_k=1e-8,
            h_0=250e-9, # initial height (halved)
            mu=1e-7,
        ),
    )
