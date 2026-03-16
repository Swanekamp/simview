"""
Self-contained tests for the plotting API.

Tests:
1) contour_plot
2) draw_structure_rz
3) make_batch_contours
4) make_gif

All tests use synthetic data and do NOT depend on CHICAGO or .p4 files.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from simview.structure import StructureRZ
from simview.contour_plot import contour_plot
from simview.draw_struct import draw_structure_bodies_debug, draw_structure_rz
from simview.draw_struct import draw_structure_rz
from simview.structure import SegmentRZ, StructureRZ

from simview.make_batch_contours import make_batch_contours
from simview.make_gif import make_gif
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTDIR = BASE_DIR / "test_output"
OUTDIR.mkdir(exist_ok=True)

# ============================================================
# Synthetic FLDS object for batch tests
# ============================================================

class FakeFLDS:
    def __init__(self, time):
        self.time = time


def fake_load_fn(file):
    """
    Pretend to load a file.
    Encode time in filename.
    """
    t = float(Path(file).stem.split("_")[-1])
    return FakeFLDS(t)


def fake_compute_fn(FLDS):
    """
    Generate synthetic field data.
    """

    r = np.linspace(0, 10, 100)
    z = np.linspace(0, 20, 200)

    R, Z = np.meshgrid(r, z)

    # moving Gaussian pulse
    z0 = 10 + 3*np.sin(FLDS.time/5)

    F = np.exp(-((Z - z0)**2 + (R - 5)**2)/10)

    return r, z, F


# ============================================================
# Test 1 — contour plot
# ============================================================

def test_contour_plot():

    r = np.linspace(0, 5, 80)
    z = np.linspace(0, 10, 120)

    R, Z = np.meshgrid(r, z)

    F = np.sin(Z/2)*np.exp(-((R-2.5)**2))

    fig, ax = contour_plot(
        z,
        r,
        F,
        title="Synthetic field t={time:.1f} ns",
        cbar_label="F",
        levels=50,
        cmap="plasma"
    )

    fig.savefig(OUTDIR / "test_contour.png", dpi=200)
    fig.savefig(OUTDIR / "test_contour.svg", dpi=200)
    plt.close(fig)

    print("Contour test written to test_contour.png")


# ============================================================
# Test 2 — structure drawing
# ============================================================

def test_structure_plot():

    segments = [

        # inner conductor
        SegmentRZ(1,0,1,10,1,1,1,1),
        SegmentRZ(1,10,2,10,1,1,1,1),
        SegmentRZ(2,10,2,0,1,1,1,1),
        SegmentRZ(2,0,1,0,1,1,1,1),

        # outer conductor
        SegmentRZ(4,0,4,10,1,2,1,2),
        SegmentRZ(4,10,5,10,1,2,1,2),
        SegmentRZ(5,10,5,0,1,2,1,2),
        SegmentRZ(5,0,4,0,1,2,1,2),
    ]

    struct = StructureRZ(segments)
    struct.find_connected_bodies()

    fig, ax = plt.subplots(figsize=(6,4))

    draw_structure_rz(ax, struct, polarity='negative', fill=False)

    ax.set_xlim(0,10)
    ax.set_ylim(0,6)

    ax.set_xlabel("z (cm)")
    ax.set_ylabel("r (cm)")

    ax.set_title("Negative Polarity Structure Test")

    fig.savefig(OUTDIR / "test_structure(neg-pol).png", dpi=200)

    plt.close(fig)
    draw_structure_rz(ax, struct, polarity='positive', fill=False)

    ax.set_xlim(0,10)
    ax.set_ylim(0,6)

    ax.set_xlabel("z (cm)")
    ax.set_ylabel("r (cm)")

    ax.set_title("Positive Polarity Structure Test")

    fig.savefig(OUTDIR / "test_structure(pos-pol).png", dpi=200)

    plt.close(fig)

    print("Structure test written to test_structure.png")


# ============================================================
# Test 3 — batch contour generation
# ============================================================

def test_batch_contours():

    files = [f"flds_{i}" for i in range(8)]

    outdir = Path("test_frames")

    make_batch_contours(
        files,
        load_fn=fake_load_fn,
        compute_fn=fake_compute_fn,
        outdir=outdir,
        plot_name="Ez",
        title="Synthetic field t={time:.1f} ns",
        cbar_label="Ez",
        cmap="viridis",
        fmt="svg"
    )

    print("Batch contour test complete")


# ============================================================
# Test 4 — GIF generation
# ============================================================

def test_gif():
    print("Starting GIF test...")
    make_gif("test_frames", OUTDIR / "test_animation.gif", fps=5)
    print("GIF test complete")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":

    test_contour_plot()

    test_structure_plot()

    test_batch_contours()

    test_gif()

    print("\nAll plotting API tests complete.")