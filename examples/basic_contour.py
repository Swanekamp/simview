"""
Basic contour plot example.

Demonstrates how to visualize a scalar field on an R-Z grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from simview.contour_plot import contour_plot

# Create synthetic grid
z = np.linspace(0, 10, 200)
r = np.linspace(0, 5, 100)

Z, R = np.meshgrid(z, r)

# Example scalar field
field = np.exp(-((R - 2.5)**2 + (Z - 5)**2) / 2)

fig, ax = contour_plot(
    z,
    r,
    field,
    title="Example Scalar Field",
    cbar_label="Field amplitude",
)

plt.show()

