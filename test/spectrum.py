import numpy as np
import matplotlib.pyplot as plt
from thin_film.light import interfere, reflectance_to_rgb


"""
Generates the thin-film interference spectrum to test the renderer.
The expected result looks like the image on this page: https://soapbubble.fandom.com/wiki/Color_and_Film_Thickness 
Although it's a lot dimmer, since idk what illuminant they used or if they handled reflectance
"""

all_wavelengths = np.linspace(380, 780, num=32) * 1e-9
h = np.arange(0, 1600, step=1) * 1e-9

reflectance = interfere(all_wavelengths, n1=1, n2=1.33, theta1=0, h=h)
rgb = reflectance_to_rgb(reflectance)
rgb = rgb[np.newaxis, :, :]

plt.imshow(rgb, aspect=500, interpolation="nearest", extent=[0, 1600, 0, 1])
plt.yticks([])
plt.xlabel("thickness (nm)")
plt.show()
