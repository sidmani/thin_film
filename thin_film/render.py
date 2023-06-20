from multiprocessing import Pool
import numpy as np
from .color_system import cs_srgb, cmf
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
from .util import _raise, init_process
from rich import print
from .fork_pdb import init_fork_pdb
from multiprocessing import Pool, Manager


def generate_sampling_coords(res, bounds):
    px, py = np.mgrid[0 : res[0] : 1, 0 : res[1] : 1]
    px = (bounds[2] - bounds[0]) * (px + 0.5) / res[0] + bounds[0]
    py = (bounds[3] - bounds[1]) * (py + 0.5) / res[1] + bounds[1]
    return np.c_[px.ravel(), py.ravel()]


# TODO: don't need 81 buckets
def spec_to_rgb(spec, T):
    # sum [batch, 81, 3] over axis 1 -> XYZ is [batch, 3]
    xyz = np.sum(spec[:, :, np.newaxis] * cmf[np.newaxis, :, :], axis=1)
    # den [batch, 1]
    xyz = xyz / xyz.sum(axis=1, keepdims=True).clip(min=0)

    rgb = np.einsum("ij,kj->ki", T, xyz)
    min_v = rgb.min()
    if min_v < 0:
        rgb -= min_v
    del xyz
    rgb /= np.max(rgb)

    # TODO: normalize
    return rgb


def fresnel(n1, n2, theta1):
    # compute reflected power
    # see https://en.wikipedia.org/wiki/Fresnel_equations
    cos_theta_i = np.cos(theta1)
    cos_theta_t = (1 - ((n1 / n2) * np.sin(theta1)) ** 2) ** 0.5

    # reflectance for s- and p-polarized waves
    R_s = (
        (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    ) ** 2
    R_p = (
        (n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i)
    ) ** 2

    # assume the light source is nonpolarized, so average the result
    return (R_s + R_p) / 2


def interfere(all_wavelengths, n1, n2, theta1, h):
    # the optical path difference of a first-order reflection
    D = 2 * n2 * h * np.cos(theta1)

    # the corresponding first-order wavelength-dependent phase shift
    phase_shift = 2 * np.pi * D[:, np.newaxis] / all_wavelengths

    # use the Fresnel equations to compute the reflected power
    r_as = fresnel(n1, n2, theta1)
    t_as = 1 - r_as

    r_sa = fresnel(n2, n1, theta1)
    t_sa = 1 - r_sa

    r = np.abs(
        r_as
        + (t_as * r_sa * t_sa * np.exp(1j * phase_shift))
        / (1 - r_sa**2 * np.exp(1j * phase_shift))
    )

    return r**2


def render_frame(args):
    r, res, constants, chunk_size = args

    sampling_coords = generate_sampling_coords(res, constants.bounds)
    kdtree = KDTree(r)

    chunks = []
    all_wavelengths = np.arange(380, 785, step=5) * 1e-9
    for i in range(0, res[0] * res[1], chunk_size):
        # interpolate the height using the SPH kernel
        (interp_h,) = get_numerical_height(
            chunk=(i, min(i + chunk_size, sampling_coords.shape[0])),
            query_pts=sampling_coords,
            kdtree=kdtree,
            constants=constants,
        )

        intensity = (
            interfere(all_wavelengths, n1=1, n2=1.33, theta1=0, h=2 * interp_h)
        )

        rgb = spec_to_rgb(intensity, cs_srgb.T)
        chunks.append(rgb)

    return np.concatenate(chunks).reshape(*res, 3)


def render(data, workers, res, constants, pixel_chunk_size):
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
                map(lambda f: (f[0], res, constants, pixel_chunk_size), data),
            ),
            description="Render",
            total=len(data),
        ):
            frames.append(frame)

    return frames
