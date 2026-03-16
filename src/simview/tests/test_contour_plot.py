import numpy as np
from p4reader import P4Reader
from simview.contour_plot import contour_plot
import os

def test_contour_plot():
    print("Starting contour_plot test")
    os.chdir('/mnt/c/Users/sswan/OneDrive/Documents/Runs/Chicago/Fisica/run38')
    FLDS = P4Reader("flds10763.p4")

    fig, ax = contour_plot(
        FLDS.r,
        FLDS.z,
        np.abs(FLDS.Er),
        title=f"$E_r$ at t={FLDS.time:.0f} ns",
        cmap="plasma",
        vmin=0,
        vmax=1500,
        levels=100,
        line_levels=10,
        cbar_label="$E_r$ (kV/cm)",
        figsize='auto',
    )

    assert fig is not None
    assert ax is not None

    print("contour_plot OK")

test_contour_plot()
plt.show()
print("All tests passed")