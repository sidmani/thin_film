from dataclasses import dataclass
import time
from .render import render_frame, generate_sampling_coords, resample_heights
import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .step import init_values, step
from rich.progress import Progress
from rich import print
from multiprocessing import Pool


@dataclass
class Parameters:
    particle_count: int
    nb_threshold: float  # the radius to include particles as neighbors
    bounds: tuple  # the rectangular boundary of the simulation in the format (x0, y0, x1, y1)
    initial_surfactant_concentration: float
    surfactant_diffusion_coefficient: float  # the coefficient in the convection-diffusion equation for the surfactant
    kernel_h: float
    stiffness: float
    alpha_k: float
    alpha_d: float
    delta_t: float
    V: float  # the half-volume of each particle
    m: float  # the particle mass
    mu: float

    # compute the rest height of the thin film if it were uniformly distributed
    @property
    def rest_height(self):
        return (
            self.V
            * self.particle_count
            / ((self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1]))
        )

    def validate(self):
        # check bounds are oriented right
        assert self.bounds[0] < self.bounds[2]
        assert self.bounds[1] < self.bounds[3]

        # check whatever needs to be positive
        assert self.particle_count > 0
        assert self.nb_threshold > 0
        assert self.delta_t > 0


def simulate(steps, constants, workers):
    print(constants)
    res = (512, 512)

    sampling_coords = generate_sampling_coords(res, constants.bounds)
    frames = []

    with Pool(
        workers, initializer=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
    ) as pool, Progress() as progress:
        start_time = time.time()
        print("Initializing fields...", end="", flush=True)

        r, u, Gamma = init_values(constants)
        adv_h = None

        print(f"done in {(time.time() - start_time):.2f}s.")
        sim_task = progress.add_task("[red]Simulate", total=steps)
        render_task = progress.add_task("[green]Render", total=steps)

        def submit_frame(frame, idx):
            frames.append((frame, idx))
            progress.update(render_task, advance=1)

        for i in range(steps):
            r, u, Gamma, adv_h = step(r, u, Gamma, adv_h, constants, pool)
            pool.apply_async(
                render_frame,
                [r, adv_h, res, sampling_coords],
                callback=lambda frame: submit_frame(frame, i),
            )
            progress.update(sim_task, advance=1)

        pool.close()
        pool.join()

    frames.sort(key=lambda x: x[1])

    def update(f):
        im1.set_data(f[0])

    im1 = plt.imshow(frames[0][0])
    plt.gca().invert_yaxis()
    ani = FuncAnimation(
        plt.gcf(),
        func=update,
        frames=frames,
        interval=30,
    )
    plt.show()
