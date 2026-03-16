"""
Overlay a simulation field with conductor geometry.
"""

import numpy as np
import matplotlib.pyplot as plt

from simview.contour_plot import contour_plot
from simview.structure import StructureRZ
from simview.draw_struct import draw_structure_rz

# synthetic grid
z = np.linspace(0, 10, 200)
r = np.linspace(0, 5, 100)

Z, R = np.meshgrid(z, r)

field = np.sin(Z) * np.exp(-R)

fig, ax = contour_plot(
    z,
    r,
    field,
    title="Field with Structure Overlay",
)

# load structure
struct = StructureRZ()

draw_structure_rz(ax, struct)

plt.show()

