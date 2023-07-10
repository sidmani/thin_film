from collections import namedtuple
from multiprocessing import Pool
import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    CloughTocher2DInterpolator,
)
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
from .color import reflectance_to_rgb
from multiprocessing import Pool, Manager


def generate_sampling_coords(res):
    px, py = np.mgrid[0 : res[0] : 1, 0 : res[1] : 1]
    px = (px + 0.5) / res[0]
    py = (py + 0.5) / res[1]
    return np.c_[px.ravel(), py.ravel()]


def fresnel(n1, n2, theta1):
    cos_theta_i = np.cos(theta1)
    # using snell's law and 1 - sin^2 = cos^2
    # TODO: this can produce complex values that aren't handled properly
    cos_theta_t = (1 - ((n1 / n2) * np.sin(theta1)) ** 2) ** 0.5

    # amplitude reflection and transmission coefficients for s- and p-polarized waves
    r_s = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    r_p = (n1 * cos_theta_t - n2 * cos_theta_i) / (n2 * cos_theta_i + n1 * cos_theta_t)
    t_s = r_s + 1
    t_p = n1 / n2 * (r_p + 1)

    # assume the light source is nonpolarized, so average the results
    return (r_s + r_p) / 2, (t_s + t_p) / 2


def interfere(all_wavelengths, n1, n2, theta1, h):
    # the optical path difference of a first-order reflection
    D = 2 * n2 * h * np.cos(theta1)

    # the corresponding first-order wavelength-dependent phase shift
    phase_shift = 2 * np.pi * D[:, np.newaxis] / all_wavelengths

    # use the Fresnel equations to compute the reflection coefficients
    r_as, t_as = fresnel(n1, n2, theta1)
    r_sa, t_sa = fresnel(n2, n1, theta1)

    # geometric sum of the complex amplitudes of all reflected waves
    # squared to yield intensity
    return (
        np.abs(
            r_as
            + (t_as * r_sa * t_sa * np.exp(1j * phase_shift))
            / (1 - r_sa**2 * np.exp(1j * phase_shift))
        )
        ** 2
    )


def render_frame(args):
    ((r, adv_h), render_args) = args

    # compute the reflectance and rgb values
    all_wavelengths = np.linspace(380, 780, num=render_args.wavelength_buckets) * 1e-9
    reflectance = interfere(all_wavelengths, n1=1, n2=1.33, theta1=0, h=2 * adv_h)
    rgb = reflectance_to_rgb(reflectance)

    if render_args.interpolation == "nearest":
        interpolate = NearestNDInterpolator(r, rgb)
    elif render_args.interpolation == "linear":
        interpolate = LinearNDInterpolator(r, rgb, fill_value=0)
    elif render_args.interpolation == "cubic":
        interpolate = CloughTocher2DInterpolator(r, rgb, fill_value=0)

    chunks = []
    sampling_coords = generate_sampling_coords(render_args.res)
    for i in range(
        0, render_args.res[0] * render_args.res[1], render_args.pixel_chunk_size
    ):
        chunk = interpolate(sampling_coords[i: min(i + render_args.pixel_chunk_size, sampling_coords.shape[0])])
        chunks.append(chunk)

    return np.concatenate(chunks).reshape(*render_args.res, 3, order="F")


RenderArgs = namedtuple(
    "RenderArgs",
    [
        "res",
        "pixel_chunk_size",
        "wavelength_buckets",
        "interpolation",
    ],
)


def render(
    data,
    workers,
    render_args,
):
    manager = Manager()
    stdin_lock = manager.Lock()
    init_fork_pdb(stdin_lock)

    frames = []
    with Pool(
        workers, initializer=init_process, initargs=[stdin_lock]
    ) as pool, Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        SpinnerColumn(),
    ) as progress:
        for frame in progress.track(
            pool.imap(
                render_frame,
                map(
                    lambda step_data: (
                        step_data,
                        render_args,
                    ),
                    data,
                ),
            ),
            description="Render",
            total=len(data),
        ):
            frames.append(frame)

    return frames
