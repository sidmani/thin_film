from .fork_pdb import fork_pdb

from multiprocessing import cpu_count
import argparse
import sys
import os
from rich import print
from .simulate import simulate, Parameters


def main():
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

    print("[red]Thin-film simulator launched.[/red]")
    print(f"Using {args.workers} workers on {cpu_count()} CPUs.")

    simulate(
        args.timesteps,
        Parameters(
            particle_count=args.particle_count,
            V=2e-11,
            m=2e-8,
            # diffusion coefficients in liquids are 1e-9 to 1e-10
            surfactant_diffusion_coefficient=1e-9,
            initial_surfactant_concentration=1e-6,
            nb_threshold=args.particle_nb_r,
            kernel_h=1.01 * args.particle_nb_r,
            delta_t=args.delta_t,
            alpha_h=1e-1,
            alpha_k=1e-1,
            alpha_d=1e-1,
            mu=1e-3,
            bounds=tuple(args.bounds),
        ),
        workers=args.workers,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted. Shutting down...")
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
