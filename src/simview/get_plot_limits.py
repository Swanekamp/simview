import numpy as np
def get_plot_limits(obj, quantity, scale=1.0, symmetric=False, step=None):
    if hasattr(obj, quantity):
        F = getattr(obj, quantity)
    else:
        F = obj[quantity]

    F = np.asarray(F) * scale
    fmin = np.nanmin(F)
    fmax = np.nanmax(F)

    if symmetric:
        bound = max(abs(fmin), abs(fmax))
        fmin, fmax = -bound, bound

    if step is not None and step > 0:
        fmin = step * np.floor(fmin / step)
        fmax = step * np.ceil(fmax / step)

    return fmin, fmax