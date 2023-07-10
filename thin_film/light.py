import numpy as np
import scipy.interpolate

# CIE 1931 color matching function
CMF = np.loadtxt("cie-cmf.txt", usecols=(1, 2, 3))

# Standard illuminant D65
std65 = np.loadtxt(
    "std65.txt",
    usecols=(
        0,
        1,
    ),
)

WAVELENGTHS = std65[:, 0] * 1e-9
D65 = std65[:, 1]

# Transformation matrix between XYZ and linear RGB coordinates
XYZ_TO_RGB = np.linalg.inv(
    np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
)


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


# compute the reflected intensity of each wavelength
def interfere(all_wavelengths, n1, n2, theta1, h):
    # the optical path difference of a first-order reflection
    D = 2 * n2 * h * np.cos(theta1)

    # the corresponding first-order wavelength-dependent phase shift
    phase_shift = 2 * np.pi * D[:, np.newaxis] / all_wavelengths

    # use the Fresnel equations to compute the reflection and transmission coefficients
    # for air-water and water-air boundaries
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


def reflectance_to_rgb(reflectance, illuminant=D65, luminance_factor=1):
    # interpolate CMF and illuminant
    buckets = reflectance.shape[1]
    if buckets < CMF.shape[0]:
        coords = np.linspace(380, 780, num=buckets) * 1e-9
        interp_CMF = scipy.interpolate.griddata(WAVELENGTHS, CMF, coords)
        interp_illuminant = np.interp(coords, WAVELENGTHS, illuminant)
    else:
        interp_CMF = CMF
        interp_illuminant = illuminant

    # integrate the color matching function
    # left side is [batch, buckets, 1] and cmf is [buckets, 3] -> result is [batch, 3]
    xyz = np.sum(
        (reflectance * interp_illuminant)[..., np.newaxis] * interp_CMF, axis=1
    )

    # normalize XYZ
    xyz /= (interp_illuminant * interp_CMF[:, 1]).sum()

    # convert XYZ to linear-rgb values
    lin_rgb = (np.einsum("ij,kj->ki", XYZ_TO_RGB, xyz) * luminance_factor).clip(
        min=0, max=1
    )

    # apply gamma correction
    return np.where(
        lin_rgb <= 0.0031308, 12.92 * lin_rgb, 1.055 * lin_rgb ** (1 / 2.4) - 0.055
    )
