"""
Overlay a simulation field with conductor geometry.
"""

import numpy as np
import matplotlib.pyplot as plt

from simview.contour_plot import contour_plot
from simview.structure import SegmentRZ, StructureRZ
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

# synthetic rectangular conductor body (mty=1 marks conductor)
segments = [
    SegmentRZ(r0=1.0, z0=3.0, r1=2.0, z1=3.0, mty=1, mid=0, nty=0, nid=0),
    SegmentRZ(r0=2.0, z0=3.0, r1=2.0, z1=7.0, mty=1, mid=0, nty=0, nid=0),
    SegmentRZ(r0=2.0, z0=7.0, r1=1.0, z1=7.0, mty=1, mid=0, nty=0, nid=0),
    SegmentRZ(r0=1.0, z0=7.0, r1=1.0, z1=3.0, mty=1, mid=0, nty=0, nid=0),
]
struct = StructureRZ(segments)

draw_structure_rz(ax, struct)

plt.show()
