import numpy as np

# CIE 1931 color matching function
CMF = np.loadtxt("cie-cmf.txt", usecols=(1, 2, 3))
# Standard illuminant D65
D65 = np.loadtxt("std65.txt", usecols=(1,))

XYZ_TO_sRGB = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, 0.2040, 1.0570]]
)


def reflectance_to_rgb(reflectance, illuminant=D65):
    # integrate the color matching function
    # left side is [batch, 81, 1] and cmf is [81, 3] -> result is [batch, 3]
    xyz = np.sum((reflectance * illuminant)[..., np.newaxis] * CMF, axis=1)

    # normalize XYZ
    illuminant_luminosity = (illuminant * CMF[:, 1]).sum()
    xyz = xyz / illuminant_luminosity

    # convert XYZ to linear-rgb values
    lin_rgb = np.einsum("ij,kj->ki", XYZ_TO_sRGB, xyz).clip(min=0, max=1)

    # apply gamma correction
    rgb = np.where(
        lin_rgb <= 0.0031308, 12.92 * lin_rgb, 1.055 * lin_rgb ** (1 / 2.4) - 0.055
    )
    return rgb
