from dataclasses import dataclass
from .step import step
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from multiprocessing import Pool, Manager
import numpy as np
from .util import init_process
from .fork_pdb import init_fork_pdb


@dataclass
class Parameters:
    particle_count: int
    nb_threshold: float  # the radius to include particles as neighbors
    initial_surfactant_concentration: float
    surfactant_diffusion_coefficient: float  # the coefficient in the convection-diffusion equation for the surfactant
    stiffness: float
    alpha_k: float
    alpha_d: float
    delta_t: float
    V: float  # the half-volume of each particle
    m: float  # the particle mass
    mu: float
    rest_height: float

    @property
    def bounds(self):
        edge = np.sqrt(self.V * self.particle_count / self.rest_height)
        return (0, 0, edge, edge)


# TODO: do a better job of initializing
def init_values(constants):
    bounds = constants.bounds
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

        for _ in progress.track(range(steps), description="Simulate"):
            r, u, Gamma, adv_h = step(r, u, Gamma, adv_h, constants, pool)
            data.append((r.copy(), adv_h.copy()))

    return data
