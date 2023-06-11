from dataclasses import dataclass
import scipy.constants
from .fork_pdb import fork_pdb
import time
from .render import render_frame, generate_sampling_coords
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count
from .step import init_values, step
import pprint


@dataclass
class Parameters:
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
    bounds: tuple


def run(steps, constants, workers):
    print("Thin-film simulator launched with parameters:")
    pprint.pprint(constants)
    res = (512, 512)

    sampling_coords = generate_sampling_coords(res, constants.bounds)

    frames = []
    with Pool(workers) as pool:
        print(f"Using {workers} workers on {cpu_count()} CPUs.")
        start_time = time.time()
        print("Initializing fields...", end="", flush=True)
        r, u, Gamma, num_h, adv_h = init_values(constants, pool)
        print(f"done in {(time.time() - start_time):.2f}s.")
        print("Entering simulation loop.")
        frame_pbar = tqdm(total=steps, position=1, leave=True, desc="Render")

        def submit_frame(frame, idx):
            frames.append((frame, idx))
            frame_pbar.update(n=1)

        for i in tqdm(range(steps), position=0, leave=True, desc="Simulate"):
            r, u, Gamma, num_h, adv_h = step(r, u, Gamma, num_h, adv_h, constants, pool)
            pool.apply_async(
                render_frame,
                [r, adv_h, res, sampling_coords],
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
        Parameters(
            particle_count=10000,
            V=1e-10,
            m=1e-10,
            gamma_a=293.15 * scipy.constants.R,
            nb_threshold=0.1,
            gamma_0=72e-3,  # surface tension of water at room temp = 72 mN/m
            delta_t=1 / 60,  # 60 fps
            alpha_c=1e-8,  # diffusion coefficient of surfactant
            alpha_d=1e-8,
            alpha_h=1e-8,
            alpha_k=1e-8,
            h_0=250e-9,  # initial height (halved)
            mu=1e-7,
            bounds=(0, 0, 1, 1),
        ),
        workers=cpu_count() - 1,
        # workers=1,
    )
