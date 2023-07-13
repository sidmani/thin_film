import argparse
import numpy as np
import sys
import os
from multiprocessing import cpu_count
from matplotlib.animation import FuncAnimation
from rich import print
from thin_film.render import RenderArgs, render
import matplotlib.pyplot as plt
from thin_film.util import exit_with_error
from .simulate import simulate, Parameters


def main():
    parser = argparse.ArgumentParser(
        prog="thinfilm",
        description="Thin film simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="the number of processes used concurrently. defaults to cpu_count - 1",
        default=cpu_count() - 1,
    )
    # arguments for simulator
    parser.add_argument("--simulate", action="store_true", help="run the simulator")

    parser.add_argument(
        "--rest-height",
        type=float,
        help="the rest height (thickness) of the thin-film in meters. usually you want a value from 1e-7 to 5e-7.",
        default=250e-9,
    )
    parser.add_argument(
        "--timesteps", type=int, help="the number of timesteps to simulate", default=10
    )
    parser.add_argument(
        "--delta-t", type=float, help="the time interval between frames", default=1 / 30
    )
    parser.add_argument(
        "--particle-count",
        type=int,
        help="the number of particles in the simulation",
        default=10000,
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        help="the resistance of the fluid to compression",
        default=10,
    )
    parser.add_argument(
        "--vorticity",
        type=float,
        help="the swirliness of the fluid",
        default=1e-2
    )
    parser.add_argument(
        "--viscosity",
        type=float,
        help="the goopiness of the fluid",
        default=1
    )

    # args for renderer
    parser.add_argument(
        "--render", action="store_true", help="render the simulated data"
    )
    parser.add_argument(
        "--res",
        type=int,
        help="the resolution of the rendered video as a single integer. Only square videos are supported.",
        default=512,
    )
    parser.add_argument(
        "--pixel-chunk",
        type=int,
        help="the number of pixels to render simultaneously per core. Higher number = faster, but more memory usage.",
        default=250000,
    )
    parser.add_argument(
        "--wavelength-buckets",
        type=int,
        help="the number of samples of the spectrum used for rendering. Higher number = slower, more memory usage, more accurate. Maximum is 81, but anything over 32 is unnoticeable",
        default=16,
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="display the rendered video using matplotlib",
    )

    # parser.add_argument(
    #     "--sim-output",
    #     help="The file to save the raw simulation data to. .hdf5 will be appended to the filename if necessary. If this argument is not provided, the data will be maintained in memory",
    # )

    args = parser.parse_args()

    print(
        f"Thin-film simulator launched using {args.workers} workers on {cpu_count()} CPUs."
    )

    constants = Parameters(
        particle_count=args.particle_count,
        # diffusion coefficients in liquids are 1e-9 to 1e-10
        surfactant_diffusion_coefficient=1e-9,
        initial_surfactant_concentration=1e-6,
        target_nb_size=300,
        delta_t=args.delta_t,
        vorticity=args.vorticity,
        stiffness=args.stiffness,
        alpha_k=1,
        alpha_d=1,
        viscosity=args.viscosity,
        rest_height=args.rest_height,
    )

    if not args.simulate and not args.render:
        exit_with_error("No commands received! Use --simulate or --render.")

    if args.simulate:
        data = simulate(workers=args.workers, steps=args.timesteps, constants=constants)

    if args.render:
        if not args.simulate:
            exit_with_error("No input provided to renderer!")

        render_args = RenderArgs(
            res=args.res,
            pixel_chunk_size=args.pixel_chunk,
            wavelength_buckets=args.wavelength_buckets,
        )

        render(
            data,
            workers=args.workers,
            constants=constants,
            render_args=render_args,
        )
        # if args.display:
        #     im1 = plt.imshow(frames[0])

        #     def update(f):
        #         im1.set_data(f)

        #     plt.gca().invert_yaxis()
        #     ani = FuncAnimation(
        #         plt.gcf(),
        #         func=update,
        #         frames=frames,
        #         interval=30,
        #     )
        #     plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted. Shutting down...")
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
