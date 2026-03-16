import os
import numpy as np
import matplotlib.pyplot as plt

from p4reader import P4Reader, P4Structure
from simview.structure import StructureRZ
from simview.contour_plot import contour_plot
from simview.draw_struct import draw_structure_bodies_debug, draw_structure_rz

def test_draw_struct():
    print("starting test_draw_struct...")
    os.chdir('/mnt/c/Users/sswan/OneDrive/Documents/Runs/Chicago/Fisica/run38')

    FLDS = P4Reader("flds10763.p4")
    STRUCT = P4Structure("struct.p4")

    struct = StructureRZ.from_p4structure(STRUCT)

    fig, ax = contour_plot(
        FLDS.z,
        FLDS.r,
        np.abs(FLDS.Er),
        title="Structure overlay test",
        cbar_label="$E_r$ (kV/cm)",
        figsize='auto'
    )

    draw_structure_rz(ax, struct, linewidth=1.5, foil_mid=1, fill=False)
    plt.show()

test_draw_struct()