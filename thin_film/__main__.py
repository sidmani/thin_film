from dataclasses import dataclass
import scipy.constants
import numpy as np
from .render import render_frame
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count
from .step import init_values, step


@dataclass
class Constants:
    particle_count: int
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
    r, u, Gamma, num_h, adv_h = init_values(constants, bounds)

    # r = np.random.rand(particle_count, 2) * np.array(
    #     [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    # ) + np.array([bounds[0], bounds[1]])
    # u = np.zeros((particle_count, 2))
    # # surfactant concentration (Γ)
    # Gamma = np.zeros((particle_count,))
    # # numerical and advected height
    # # TODO: num_h and adv_h are off by a few orders of magnitude
    # num_h = np.ones((particle_count,)) * constants.h_0
    # adv_h = np.ones((particle_count,)) * constants.h_0

    frames = []
    frame_pbar = tqdm(total=steps, position=1, leave=True, desc="Render")

    def submit_frame(frame, idx):
        frames.append((frame, idx))
        frame_pbar.update(n=1)

    with Pool(workers) as pool:
        for i in tqdm(range(steps), position=0, leave=True, desc="Simulate"):
            r, u, Gamma, num_h, adv_h = step(r, u, Gamma, num_h, adv_h, constants, pool)
            pool.apply_async(
                render_frame,
                [r, adv_h, (512, 512), bounds],
                callback=lambda frame: submit_frame(frame, i),
            )

        pool.close()
        pool.join()

    frames.sort(key=lambda x: x[1])
    frames = list(map(lambda x: x[0], frames))

    im1 = plt.imshow(frames[0])
    ani = FuncAnimation(
        plt.gcf(),
        func=lambda f: im1.set_data(f),
        frames=frames,
        interval=200,
    )
    plt.show()


if __name__ == "__main__":
    run(
        30,
        Constants(
            particle_count=10000,
            V=1e-11,
            m=1e-8,
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
        # workers=cpu_count() - 1,
        workers=1,
    )
