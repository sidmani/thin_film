import numpy as np

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


def interp_multicol(x, xp, fp):
    results = []
    for i in range(fp.shape[1]):
        results.append(np.interp(x, xp, fp[:, i]))
    return np.column_stack(results)


def reflectance_to_rgb(reflectance, illuminant=D65, luminance_factor=1):
    # interpolate CMF and illuminant
    buckets = reflectance.shape[1]
    if buckets < CMF.shape[0]:
        coords = np.linspace(380, 780, num=buckets) * 1e-9
        interp_CMF = interp_multicol(coords, WAVELENGTHS, CMF)
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
    illuminant_luminosity = (interp_illuminant * interp_CMF[:, 1]).sum()
    xyz = xyz / illuminant_luminosity

    # convert XYZ to linear-rgb values
    lin_rgb = (np.einsum("ij,kj->ki", XYZ_TO_RGB, xyz) * luminance_factor).clip(
        min=0, max=1
    )

    # apply gamma correction
    rgb = np.where(
        lin_rgb <= 0.0031308, 12.92 * lin_rgb, 1.055 * lin_rgb ** (1 / 2.4) - 0.055
    )
    return rgb
