# simview

simview is a lightweight visualization toolkit for
2-D axisymmetric plasma and pulsed-power simulations.

It operates on NumPy arrays and can therefore be used
with data from many simulation tools (Chicago, WarpX,
LSP, VSim, ICEPIC, custom codes, etc.).

While commonly used together with the companion
reader library, simview itself is completely independent of the 
file format.

It builds on top of the basic data reader and provides utilities for:

- plotting field contours
- visualizing diode structures
- generating batches of frames from simulation dumps
- building animations from those frames
- applying smoothing filters to field data

The goal is to make it easy to go from raw `.p4` files to
publication-quality plots or diagnostic movies.

---

# Features

### Field visualization

Create contour plots of scalar or vector components on the simulation grid.

- automatic axis labeling
- configurable color scales
- optional contour lines
- publication-ready layout

---

### Structure visualization

Plot conductor boundaries and automatically identify connected bodies
such as cathodes and anodes.

Utilities include:

- connected-body detection
- electrode classification
- structure overlays for field plots

---

### Batch plotting

Generate a sequence of plots from a series of field dumps.

Typical use case:


flds0001.p4 \
flds0002.p4 \
flds0003.p4 \
... 


Produce:


Jz_001ns.png \
Jz_002ns.png \
Jz_003ns.png \
...


---

### Animation tools

Convert generated frames into GIF animations for quick visualization of simulation dynamics.

---

### Smoothing filters

Reusable smoothing functions for simulation fields:

- Gaussian filters
- box filters
- Savitzky–Golay smoothing
- simple 3-point and 5-point smoothing

Useful for reducing PIC noise in diagnostics.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/swanekamp/simview.git
cd simview
```
Install the package and make it editable:
```bash
pip install -e .
```

# Requirements:

numpy
matplotlib
imageio

### Optional (for smoothing filters):

scipy

# Examples

## Basic contour plot example
```bash
cd sim_directory
python basic_contour.py
```
```python
## Example 1 — Basic contour plot

examples/basic_contour.py

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

# Example field
field = np.exp(-((R - 2.5)**2 + (Z - 5)**2) / 2)

fig, ax = contour_plot(
    z,
    r,
    field,
    title="Example Scalar Field",
    cbar_label="Field amplitude",
)

plt.show()
```
## Other Examples: 
`examples/basic_contour.py`        – simple field visualization \
`examples/field_with_structure.py` - plot field quantity with structure overlay \
`examples/batch_contours.py`       – batch frame generation \
`examples/create_animation.py`     – build GIF animation \
`examples/run_simview_chicago.py`  – advanced example demonstrating how to use simview to visualize Chicago simulation data and create animations (requires p4reader)
