import numpy as np

cmf = np.loadtxt("cie-cmf.txt", usecols=(1, 2, 3))


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1 - x - y))


class ColorSystem:
    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals

    def __init__(self, red, green, blue, white):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    # def xyz_to_rgb(self, xyz, out_fmt=None):
    #     rgb = self.T.dot(xyz)
    #     if np.any(rgb < 0):
    #         # We're not in the RGB gamut: approximate by desaturating
    #         w = - np.min(rgb)
    #         rgb += w
    #     if not np.all(rgb==0):
    #         # Normalize the rgb vector
    #         rgb /= np.max(rgb)

    #     if out_fmt == 'html':
    #         return self.rgb_to_hex(rgb)
    #     return rgb

    # def spec_to_xyz(self, spec):
    #     """Convert a spectrum to an xyz point.

    #     The spectrum must be on the same grid of points as the colour-matching
    #     function, self.cmf: 380-780 nm in 5 nm steps.

    #     """

    #     # spec has shape [81]; cmf: [81, 3]
    #     XYZ = np.sum(spec[:, np.newaxis] * cmf, axis=0)
    #     # XYZ [3]
    #     den = np.sum(XYZ)
    #     if den == 0.:
    #         return XYZ
    #     return XYZ / den


illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_srgb = ColorSystem(
    red=xyz_from_xy(0.64, 0.33),
    green=xyz_from_xy(0.30, 0.60),
    blue=xyz_from_xy(0.15, 0.06),
    white=illuminant_D65,
)
