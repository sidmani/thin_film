from dataclasses import dataclass
import pdb
import scipy.constants
import taichi as ti
import numpy as np
from .kernel import grad_W_spiky, W_spline4
from .render import render_frame
from tqdm import tqdm
import thin_film.physics as physics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ti.init(arch=ti.cpu)


# @ti.dataclass
@dataclass
class Constants:
    gamma_0: float
    gamma_a: float
    nb_threshold: float
    alpha_c: float
    alpha_d: float
    alpha_h: float
    alpha_k: float
    delta_t: float
    h_0: float
    V: float
    mu: float
    m: float


# @ti.kernel
def step(
    r: ti.types.ndarray(),
    u: ti.types.ndarray(),
    Gamma: ti.types.ndarray(),
    num_h: ti.types.ndarray(),
    adv_h: ti.types.ndarray(),
    constants: Constants,
):
    divergence = np.empty_like(Gamma)
    curvature = np.empty_like(Gamma)
    pressure = np.empty_like(Gamma)
    new_Gamma = np.empty_like(Gamma)

    surface_tension = physics.compute_surface_tension(
        constants.gamma_0, constants.gamma_a, Gamma
    )

    for i in tqdm(range(r.shape[0]), position=1, leave=False):
        rij = r - r[i]
        # compute the neighborhood
        r_len = np.sqrt(np.sum(rij**2, axis=1))
        nb = (r_len < constants.nb_threshold) & (r_len > 0)

        grad_kernel = grad_W_spiky(rij[nb], constants.nb_threshold, r_len[nb])
        grad_kernel_reduced = 2 * np.sqrt(np.sum(grad_kernel**2, axis=1)) / r_len[nb]

        divergence[i] = physics.compute_divergence(
            constants.V, num_h[nb], u[nb] - u[i], grad_kernel
        )
        curvature[i] = physics.compute_curvature(
            constants.V, num_h[nb], num_h[i], grad_kernel_reduced
        )
        new_Gamma[i] = physics.compute_surfactant_diffusion(
            constants.V,
            num_h[nb],
            Gamma[nb],
            Gamma[i],
            constants.alpha_c,
            constants.delta_t,
            grad_kernel_reduced,
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
    adv_h += -adv_h * divergence * constants.delta_t

    force = np.zeros_like(r)
    new_num_h = np.empty_like(num_h)

    # compute forces
    for i in tqdm(range(r.shape[0]), position=1, leave=False):
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
            constants.V, num_h[nb], surface_tension[nb], num_h[i], surface_tension[i], grad_kernel
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

    # TODO: updating by half should improve stability
    # 5. update velocity
    u += constants.delta_t / constants.m * force
    # 6. update position
    r += u * constants.delta_t

    return r, u, new_Gamma, new_num_h, adv_h



def run(particle_count, steps, constants):
    bounds = (0, 0, 1, 1)
    frames = []

    r = np.random.rand(particle_count, 2) * np.array(
        [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    ) + np.array([bounds[0], bounds[1]])
    u = np.zeros((particle_count, 2))
    # surfactant concentration (Γ)
    Gamma = np.zeros((particle_count, 1))
    # numerical and advected height
    num_h = np.ones((particle_count, 1)) * constants.h_0
    adv_h = np.ones((particle_count, 1)) * constants.h_0

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
        10000,
        20,
        Constants(
            V=1e-7,
            m=1e-7,
            nb_threshold=0.01,
            gamma_0=72e-3,  # surface tension of water at room temp = 72 mN/m
            gamma_a=scipy.constants.R * 293.15,
            delta_t=1 / 60,  # 60 fps
            alpha_c=1e-8,  # diffusion coefficient of surfactant
            alpha_d=1e-8,
            alpha_h=1e-8,
            alpha_k=1e-8,
            h_0=250e-9,  # initial height (halved)
            mu=1e-7,
        ),
    )
