"""
Example demonstrating batch generation of contour plots.

This script creates a sequence of synthetic fields and saves
each frame to disk. The resulting images can be used to build
an animation.
"""

import numpy as np
from simview.make_batch_contours import make_batch_contours


# -------------------------------------------------
# Synthetic data container
# -------------------------------------------------

class Frame:
    def __init__(self, z, r, field, time):
        self.z = z
        self.r = r
        self.field = field
        self.time = time


# -------------------------------------------------
# Synthetic data loader
# -------------------------------------------------

def load_field(step):
    """
    Return a Frame with z, r, field, and time for a given step.
    """

    z = np.linspace(0, 10, 200)
    r = np.linspace(0, 5, 100)

    Z, R = np.meshgrid(z, r)

    field = np.sin(Z - 0.3 * step) * np.exp(-R)

    return Frame(z=z, r=r, field=field, time=step)


# -------------------------------------------------
# Field computation function
# -------------------------------------------------

def compute_field(data):
    """
    In a real workflow this could compute Jz, Ez, etc.
    Here we just return the synthetic field.
    """
    return data.r, data.z, data.field


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
