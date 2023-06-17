from dataclasses import dataclass
from .render import render_frame, generate_sampling_coords, resample_heights
import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .step import init_values, step
from rich.progress import Progress, MofNCompleteColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import print
from multiprocessing import Pool, Manager
from .fork_pdb import init_fork_pdb


@dataclass
class Parameters:
    particle_count: int
    nb_threshold: float  # the radius to include particles as neighbors
    # TODO: it's way more meaningful to set the desired thickness and then determine the square boundary from that
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
    # TODO: does this make sense numerically considering how num_h is calculated?
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

def init_process(stdin_lock):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    init_fork_pdb(stdin_lock)

def _raise(e):
    raise e

def simulate(steps, constants, workers):
    print(constants)
    res = (512, 512)

    sampling_coords = generate_sampling_coords(res, constants.bounds)
    frames = []

    manager = Manager()
    stdin_lock = manager.Lock()
    init_fork_pdb(stdin_lock)

    with Pool(
        workers, initializer=init_process, initargs=[stdin_lock]
    ) as pool, Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        auto_refresh=False,
    ) as progress:
        r, u, Gamma = init_values(constants)
        adv_h = None
        print("Starting simulation.")
        sim_task = progress.add_task("[red]Simulate", total=steps)
        render_task = progress.add_task("[green]Render", total=steps)

        def submit_frame(frame, idx):
            frames.append((frame, idx))
            progress.update(render_task, advance=1)
            stdin_lock.acquire()
            progress.refresh()
            stdin_lock.release()

        for i in range(steps):
            r, u, Gamma, adv_h = step(r, u, Gamma, adv_h, constants, pool)
            pool.apply_async(
                render_frame,
                [r, adv_h, res, sampling_coords],
                callback=lambda frame: submit_frame(frame, i),
                error_callback=lambda e: _raise(e)
            )
            progress.update(sim_task, advance=1)
            stdin_lock.acquire()
            progress.refresh()
            stdin_lock.release()

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
