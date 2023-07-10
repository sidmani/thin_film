from collections import namedtuple
from multiprocessing import Pool
import numpy as np
from .fork_pdb import set_trace
from sklearn.neighbors import KDTree
from .step import get_numerical_height
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    SpinnerColumn,
)
from .util import init_process
from .fork_pdb import init_fork_pdb
from .light import reflectance_to_rgb, interfere
from multiprocessing import Pool, Manager


def generate_sampling_coords(res):
    px, py = np.mgrid[0:res:1, 0:res:1]
    px = (px + 0.5) / res
    py = (py + 0.5) / res
    return np.c_[px.ravel(), py.ravel()]


def render_frame(args):
    ((r,), constants, render_args, sampling_coords, all_wavelengths) = args
    kdtree = KDTree(r)

    chunks = []
    for i in range(0, render_args.res**2, render_args.pixel_chunk_size):
        chunk = (i, min(i + render_args.pixel_chunk_size, sampling_coords.shape[0]))

        # interpolate the height using the SPH kernel
        (interp_h,) = get_numerical_height(
            chunk=chunk,
            query_pts=sampling_coords,
            kdtree=kdtree,
            constants=constants,
        )

        reflectance = interfere(
            all_wavelengths, n1=1, n2=1.33, theta1=0, h=2 * interp_h
        )

        chunks.append(reflectance_to_rgb(reflectance))

    return np.concatenate(chunks).reshape(
        render_args.res, render_args.res, 3, order="F"
    )


RenderArgs = namedtuple(
    "RenderArgs",
    [
        "res",
        "pixel_chunk_size",
        "wavelength_buckets",
    ],
)


def render(
    data,
    workers,
    constants,
    render_args,
):
    manager = Manager()
    stdin_lock = manager.Lock()
    init_fork_pdb(stdin_lock)
    sampling_coords = generate_sampling_coords(render_args.res)
    all_wavelengths = np.linspace(380, 780, num=render_args.wavelength_buckets) * 1e-9

    frames = []
    with Pool(
        workers, initializer=init_process, initargs=[stdin_lock]
    ) as pool, Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        SpinnerColumn(),
        # auto_refresh=False,
    ) as progress:
        for frame in progress.track(
            pool.imap(
                render_frame,
                map(
                    lambda step_data: (
                        step_data,
                        constants,
                        render_args,
                        sampling_coords,
                        all_wavelengths,
                    ),
                    data,
                ),
            ),
            description="Render",
            total=len(data),
        ):
            frames.append(frame)

    return frames
