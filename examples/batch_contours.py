"""
Example demonstrating batch generation of contour plots.

This script creates a sequence of synthetic fields and saves
each frame to disk. The resulting images can be used to build
an animation.
"""

import numpy as np
from simview.make_batch_contours import make_batch_contours


# -------------------------------------------------
# Synthetic data loader
# -------------------------------------------------

def load_field(step):
    """
    Return z, r, field for a given time step.
    """

    z = np.linspace(0, 10, 200)
    r = np.linspace(0, 5, 100)

    Z, R = np.meshgrid(z, r)

    field = np.sin(Z - 0.3 * step) * np.exp(-R)

    return z, r, field


# -------------------------------------------------
# Field computation function
# -------------------------------------------------

def compute_field(z, r, field):
    """
    In a real workflow this could compute Jz, Ez, etc.
    Here we just return the synthetic field.
    """
    return field


# -------------------------------------------------
# Run batch generation
# -------------------------------------------------

if __name__ == "__main__":

    steps = range(20) # Placeholder. Replace with a list of simulation files (e.g. sorted(glob("flds*.p4")))

    make_batch_contours(
        files=steps,
        load_fn=load_field,
        compute_fn=compute_field,
        outdir="frames",
        plot_name="synthetic_field",
        cmap="inferno"
    )
