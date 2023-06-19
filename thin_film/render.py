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
    TimeRemainingColumn,
)
from .util import _raise, init_process
from rich import print
from .fork_pdb import init_fork_pdb
from multiprocessing import Pool, Manager

# TODO: memory usage here is pretty high
# because we're processing the frames all at once. makes more sense to chunk them
# into manageable pieces since the pixels are independent


# generate the sampling coordinates once
def generate_sampling_coords(res, bounds):
    px, py = np.mgrid[0 : res[0] : 1, 0 : res[1] : 1]
    px = (bounds[2] - bounds[0]) * (px + 0.5) / res[0] + bounds[0]
    py = (bounds[3] - bounds[1]) * (py + 0.5) / res[1] + bounds[1]
    return np.c_[px.ravel(), py.ravel()]


# compute wavelength-dependent amplitudes
def interfere(wavelength, n1, n2, theta1, d):
    # compute optical path difference
    sin_theta2 = n1 / n2 * np.sin(theta1)
    cos_theta2 = np.sqrt(1 - sin_theta2**2)
    del sin_theta2

    opd = 2 * d * n2 * cos_theta2
    del cos_theta2

    # phase difference, including the half turn added by reflection if necessary
    phase_1 = 0
    if n1 < n2:
        phase_1 = np.pi

    phase_2 = np.pi * 2 * opd[:, np.newaxis] / wavelength
    del opd
    phase_diff = np.abs(phase_1 - phase_2)
    del phase_2

    # return the new amplitude
    return 2 * np.cos(phase_diff / 2)


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


def render_frame_(args):
    # def render_frame_(r, res, constants, chunk_size):
    # wanted to use pool.imap, which doesn't unpack args
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
            interfere(all_wavelengths, n1=1, n2=1.33, theta1=0, d=2 * interp_h) ** 2
        )

        rgb = spec_to_rgb(intensity, cs_srgb.T)
        chunks.append(rgb)

    return np.concatenate(chunks).reshape(*res, 3)


def render(data, workers, res, constants, pixel_chunk_size=100000):
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
        TimeRemainingColumn(),
        auto_refresh=False,
    ) as progress:
        render_task = progress.add_task("Render", total=len(data))

        args = map(lambda f: (f[0], res, constants, pixel_chunk_size), data)
        for frame in pool.imap(render_frame_, args):
            frames.append(frame)
            progress.update(render_task, advance=1, refresh=True)

    return frames
