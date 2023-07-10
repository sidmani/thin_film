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
    target_nb_size: int # desired approximate neighborhood size
    initial_surfactant_concentration: float
    surfactant_diffusion_coefficient: float  # the coefficient in the convection-diffusion equation for the surfactant
    stiffness: float
    alpha_k: float
    alpha_d: float
    delta_t: float
    vorticity: float
    viscosity: float
    rest_height: float

    def __post_init__(self):
        # the volume of each particle
        self.V = self.rest_height / self.particle_count
        # pi r^2 / area (1) = target_nb_size / particle_count
        # self.nb_threshold = np.sqrt(self.target_nb_size / (self.particle_count * np.pi))
        self.nb_threshold = 0.1
        # particle mass is density of water (1000 kg/m^3) * particle volume
        self.m = 1000 * self.V


# TODO: do a better job of initializing
def init_values(constants):
    r = np.random.rand(constants.particle_count, 2)
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
        auto_refresh=False,
    ) as progress:
        r, u, Gamma = init_values(constants)

        for _ in progress.track(range(steps), description="Simulate"):
            r, u, Gamma = step(r, u, Gamma, constants, pool)
            data.append((r.copy(),))

    return data
