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
    mu: float
    rest_height: float

    @property
    def nb_threshold(self):
        # pi r^2 / area (1) = target_nb_size / particle_count
        return np.sqrt(self.target_nb_size / (self.particle_count * np.pi))

    @property
    def V(self):
        # the volume of each particle
        return self.rest_height / self.particle_count
    
    @property
    def m(self):
        # mass is 1000 kg/m^3 * particle volume
        return 1000 * self.V


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
        adv_h = None

        for _ in progress.track(range(steps), description="Simulate"):
            r, u, Gamma, adv_h = step(r, u, Gamma, adv_h, constants, pool)
            data.append((r.copy(), adv_h.copy()))

    return data
