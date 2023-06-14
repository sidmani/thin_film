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
import argparse


@dataclass
class Parameters:
    particle_count: int
    nb_threshold: float  # the radius to include particles as neighbors
    bounds: tuple  # the rectangular boundary of the simulation in the format (x0, y0, x1, y1)
    initial_surfactant_concentration: float
    pure_surface_tension: float  # the untainted surface tension (e.g. of pure water)
    surfactant_diffusion_coefficient: float  # the coefficient in the convection-diffusion equation for the surfactant
    alpha_h: float
    alpha_k: float
    alpha_d: float
    delta_t: float
    h_0: float
    V: float  # the half-volume of each particle
    m: float  # the particle mass
    mu: float


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
        interval=30,
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="thinfilm", description="Thin film simulator")

    subparsers = parser.add_subparsers()
    parser_simulate = subparsers.add_parser("simulate", help="run the simulator")

    parser_simulate.add_argument(
        "--workers",
        type=int,
        help="the number of processes used concurrently. defaults to cpu_count - 1",
        default=cpu_count() - 1,
    )
    parser_simulate.add_argument(
        "--timesteps", type=int, help="the number of timesteps to simulate", default=10
    )

    parser_simulate.add_argument(
        "--delta-t", type=int, help="the time interval between frames", default=1 / 30
    )

    parser_simulate.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        help="the bottom-left and top-right coordinates of the boundary rectangle in the format x0 y0 x1 y1",
        default=[0, 0, 1, 1],
    )

    parser_simulate.add_argument(
        "--particle-count",
        type=int,
        help="the number of particles in the simulation",
        default=10000,
    )
    parser_simulate.add_argument(
        "--particle-volume", type=float, help="the volume of each particle in m^3"
    )
    parser_simulate.add_argument(
        "--particle-mass", type=float, help="the mass of each particle in kg"
    )

    parser_simulate.add_argument(
        "--particle-nb-r",
        type=float,
        help="the radius of the neighborhood around each particle",
        default=0.1,
    )

    args = parser.parse_args()

    run(
        args.timesteps,
        Parameters(
            particle_count=args.particle_count,
            V=2e-11,
            m=2e-8,
            # diffusion coeffcients in liquids are 1e-9 to 1e-10
            surfactant_diffusion_coefficient=1e-9,
            initial_surfactant_concentration=1e-6,
            nb_threshold=args.particle_nb_r,
            delta_t=args.delta_t,
            alpha_d=1e-4,
            alpha_h=1e-4,
            alpha_k=1e-4,
            h_0=250e-9,
            mu=1e-5,
            bounds=tuple(args.bounds),
        ),
        workers=args.workers,
    )
