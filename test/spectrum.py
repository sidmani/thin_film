import numpy as np
import matplotlib.pyplot as plt
from thin_film.render import interfere, spec_to_rgb
from thin_film.color_system import cs_srgb

"""
Generates the thin-film interference spectrum to test the renderer.
The expected result looks like the image on this page: https://soapbubble.fandom.com/wiki/Color_and_Film_Thickness 
"""

all_wavelengths = np.arange(380, 785, step=5) * 1e-9
h = np.arange(0, 1600, step=5) * 1e-9
intensity = interfere(all_wavelengths, n1=1, n2=1.33, theta1=0, h=h)
rgb = spec_to_rgb(intensity, cs_srgb.T)[np.newaxis, :, :]

plt.imshow(rgb, aspect=500, interpolation="nearest", extent=[0, 1600, 0, 1])
plt.yticks([])
plt.xlabel("thickness (nm)")
plt.show()
