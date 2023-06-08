import numpy as np
from matplotlib import pyplot as plt
from .color_system import cs_srgb

# the difference in meters between the paths of light reflecting off the upper and lower surfaces
# taking refractive index into account
def optical_path_diff(n1, n2, theta1, d):
    sin_theta2 = n1 / n2 * np.sin(theta1)
    cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)
    return 2 * d * n2 * cos_theta2

# the phase difference, including the half turn added by reflection if necessary
def phase_diff(wavelength, n1, n2, theta1, d):
    # 2D (depth by pixels)
    opd = optical_path_diff(n1, n2, theta1, d)

    phase_1 = 0
    if n1 < n2:
        phase_1 = np.pi

    phase_2 = np.pi * 2 * opd[:, :, np.newaxis] / wavelength

    return np.abs(phase_1 - phase_2)

# the resultant amplitude
def interfere(wavelength, n1, n2, theta1, d):
    p = phase_diff(wavelength, n1, n2, theta1, d)
    return 2 * np.cos(p / 2)


E = 512
# in nanometers
fig, ax = plt.subplots(2, 2)

d = np.repeat(np.linspace(200, 1200, num=E)[:, np.newaxis], E, axis=1)
all_wavelengths = np.arange(380, 785, step=5)
amplitudes = interfere(all_wavelengths, 1, 1.33, np.pi / 4, d)

color = np.zeros((E, E, 3), dtype=float)
for i in range(E):
    for j in range(E):
        color[i, j] = cs_srgb.spec_to_rgb(amplitudes[i, j] ** 2)

ax[0, 0].imshow(d)
ax[0, 1].imshow(color)
plt.show()