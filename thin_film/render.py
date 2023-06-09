import numpy as np
import pdb
import scipy
from .color_system import cs_srgb, cmf


# sample the height values on a grid
def resample_heights(r, adv_h, res, bounds):
    # generate coordinates
    # TODO: can do this once instead of every frame
    px, py = np.mgrid[0 : res[0] : 1, 0 : res[1] : 1]
    px = (bounds[2] - bounds[0]) * px / res[0] + bounds[0]
    py = (bounds[3] - bounds[1]) * py / res[1] + bounds[1]
    points = np.c_[px.ravel(), py.ravel()]

    # sample the grid
    # TODO: try out different interpolation methods
    interp_h = scipy.interpolate.griddata(
        r, adv_h[:, None], points, method="cubic", fill_value=0
    )

    # reshape into a grid
    # TODO: check that this doesn't flip axes
    return interp_h.reshape(res)


# compute wavelength-dependent amplitudes
def interfere(wavelength, n1, n2, theta1, d):
    # compute optical path difference
    sin_theta2 = n1 / n2 * np.sin(theta1)
    cos_theta2 = np.sqrt(1 - sin_theta2**2)
    opd = 2 * d * n2 * cos_theta2

    # phase difference, including the half turn added by reflection if necessary
    phase_1 = 0
    if n1 < n2:
        phase_1 = np.pi

    phase_2 = np.pi * 2 * opd[:, :, np.newaxis] / wavelength
    phase_diff = np.abs(phase_1 - phase_2)

    # return the new amplitude
    return 2 * np.cos(phase_diff / 2)


def spec_to_rgb(spec, T):
    # sum [batch, 81, 3] over axis 1 -> XYZ is [batch, 3]
    xyz = np.sum(spec[:, :, np.newaxis] * cmf[np.newaxis, :, :], axis=1)
    # den [batch, 1]
    den = np.sum(xyz, axis=1, keepdims=True)
    xyz = xyz / den

    rgb = np.einsum("ij,kj->ki", T, xyz)
    rgb = np.clip(rgb, 0, None)

    # TODO: normalize
    return rgb


def render_frame(r, adv_h, res, bounds):
    heights = resample_heights(r, adv_h, res, bounds)
    # wavelengths of visible light in meters
    all_wavelengths = np.arange(380, 785, step=5) * 1e-9

    # using refractive index of air (1) and water (1.33)
    # looking at the film orthogonally (theta1=0)
    amplitudes = interfere(all_wavelengths, n1=1, n2=1.33, theta1=0, d=2 * heights)

    # TODO: can do this without reshaping
    amplitudes = np.reshape(amplitudes, (-1, 81))
    rgb = spec_to_rgb(amplitudes**2, cs_srgb.T)
    rgb = np.reshape(rgb, (*res, 3))
    return rgb
