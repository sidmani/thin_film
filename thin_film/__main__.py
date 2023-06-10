from dataclasses import dataclass
import scipy.constants
import numpy as np
from .render import render_frame
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from .step import step


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


def run(particle_count, steps, constants, workers):
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
    frame_results = []
    with Pool(workers) as pool:
        for i in tqdm(range(steps), position=0, leave=False):
            r, u, Gamma, num_h, adv_h = step(r, u, Gamma, num_h, adv_h, constants, pool)
            fr = pool.apply_async(render_frame, [r, adv_h, (1024, 1024), bounds], callback=lambda frame: frames.append(frame))
            frame_results.append(fr)

        # wait for frame rendering to complete
        for fr in tqdm(frame_results):
            fr.wait()

    im1 = plt.imshow(frames[0])
    def update(i):
        im1.set_data(frames[i])
    ani = FuncAnimation(plt.gcf(), update, interval=200)
    plt.show()


if __name__ == "__main__":
    run(
        10000,
        50,
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
        workers=23,
    )
