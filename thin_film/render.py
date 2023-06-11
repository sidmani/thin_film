import numpy as np
import pdb
import scipy
from .color_system import cs_srgb, cmf

# TODO: memory usage here is pretty high
# because we're processing the frames all at once. makes more sense to chunk them
# into manageable pieces since the pixels are independent

# generate the sampling coordinates once
def generate_sampling_coords(res, bounds):
    px, py = np.mgrid[0: res[0]: 1, 0: res[1]: 1]
    px = (bounds[2] - bounds[0]) * px / res[0] + bounds[0]
    py = (bounds[3] - bounds[1]) * py / res[1] + bounds[1]
    return np.c_[px.ravel(), py.ravel()]

def resample_heights(r, adv_h, res, sampling_coords):
    # sample the grid
    # TODO: try out different interpolation methods
    interp_h = scipy.interpolate.griddata(
        r, adv_h[:, None], sampling_coords, method="cubic", fill_value=0
    )

    # reshape into a grid
    # TODO: check that this doesn't flip axes
    return interp_h.reshape(res)


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

    phase_2 = np.pi * 2 * opd[:, :, np.newaxis] / wavelength
    del opd
    phase_diff = np.abs(phase_1 - phase_2)
    del phase_2

    # return the new amplitude
    return 2 * np.cos(phase_diff / 2)


def spec_to_rgb(spec, T):
    # sum [batch, 81, 3] over axis 1 -> XYZ is [batch, 3]
    xyz = np.sum(spec[:, :, np.newaxis] * cmf[np.newaxis, :, :], axis=1)
    # den [batch, 1]
    den = np.sum(xyz, axis=1, keepdims=True)
    xyz = xyz / den
    del den

    rgb = np.einsum("ij,kj->ki", T, xyz)
    # rgb = T @ xyz.T
    del xyz
    rgb = np.clip(rgb, 0, None)

    # TODO: normalize
    return rgb


def render_frame(r, adv_h, res, bounds):
    heights = resample_heights(r, adv_h, res, bounds)
    # wavelengths of visible light in meters
    all_wavelengths = np.arange(380, 785, step=5) * 1e-9

    # using refractive index of air (1) and water (1.33)
    # looking at the film orthogonally (theta1=0)
    intensity = interfere(all_wavelengths, n1=1, n2=1.33, theta1=0, d=2 * heights) ** 2
    intensity = np.reshape(intensity, (-1, 81))
    del heights
    del all_wavelengths

    rgb = spec_to_rgb(intensity, cs_srgb.T)
    del intensity 
    rgb = np.reshape(rgb, (*res, 3))
    return rgb
