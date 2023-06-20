from dataclasses import dataclass
from .step import step
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich import print
from multiprocessing import Pool, Manager
import numpy as np
from .util import _raise, init_process
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


def init_values(constants):
    bounds = constants.bounds
    # r_sqrt = np.sqrt(constants.particle_count)
    # r = generate_sampling_coords((r_sqrt, r_sqrt), constants.bounds)
    r = np.random.rand(constants.particle_count, 2) * np.array(
        [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    ) + np.array([bounds[0], bounds[1]])

    u = np.zeros_like(r)

    # surfactant concentration (Γ)
    Gamma = (
        np.random.rand(constants.particle_count)
        # * 0.001
        * constants.initial_surfactant_concentration
        # + constants.initial_surfactant_concentration * 0.9995
    )

    return r, u, Gamma


def simulate(workers, steps, constants):
    manager = Manager()
    stdin_lock = manager.Lock()
    init_fork_pdb(stdin_lock)

    data = []
    with Pool(
        workers, initializer=init_process, initargs=[stdin_lock]
    ) as pool, Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        r, u, Gamma = init_values(constants)
        adv_h = None

        for i in progress.track(range(steps), description="Simulate"):
            r, u, Gamma, adv_h = step(r, u, Gamma, adv_h, constants, pool)
            data.append((r.copy(), adv_h.copy()))

        pool.close()
        pool.join()

    return data
