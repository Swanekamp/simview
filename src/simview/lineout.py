import numpy as np


def extract_lineout(z, r, F, *, axis="r", value=None, index=None):
    """
    Extract a 1D lineout from a 2D field.

    Parameters
    ----------
    z, r : 1D arrays
    F    : 2D array, shape (len(r), len(z)) or (len(z), len(r))
    axis : "r" or "z"
        axis="r" -> return F(r) at fixed z
        axis="z" -> return F(z) at fixed r
    value : float, optional
        physical coordinate of the fixed slice
    index : int, optional
        grid index of the fixed slice

    Returns
    -------
    x : 1D array
        coordinate along the plotted axis
    y : 1D array
        extracted lineout
    """
    if F.shape == (len(z), len(r)):
        F_plot = F.T
    elif F.shape == (len(r), len(z)):
        F_plot = F
    else:
        raise ValueError(
            f"Bad field shape {F.shape}; expected {(len(r), len(z))} or {(len(z), len(r))}"
        )

    if axis == "r":
        if value is not None:
            index = int(np.argmin(np.abs(z - value)))
        elif index is None:
            raise ValueError("For axis='r', provide value or index.")
        return r, F_plot[:, index]

    elif axis == "z":
        if value is not None:
            index = int(np.argmin(np.abs(r - value)))
        elif index is None:
            raise ValueError("For axis='z', provide value or index.")
        return z, F_plot[index, :]

    else:
        raise ValueError("axis must be 'r' or 'z'")


def smooth_1d(y, method=None, kwargs=None):
    if kwargs is None:
        kwargs = {}

    y = np.asarray(y, dtype=float)

    if method is None:
        return y

    if method == "moving_average":
        window = int(kwargs.get("window", 5))
        if window < 1:
            raise ValueError("moving_average window must be >= 1")
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode="same")

    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        sigma = float(kwargs.get("sigma", 1.0))
        return gaussian_filter1d(y, sigma=sigma)

    elif method == "savgol":
        from scipy.signal import savgol_filter
        window_length = int(kwargs.get("window_length", 7))
        polyorder = int(kwargs.get("polyorder", 2))
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(y, window_length=window_length, polyorder=polyorder)

    else:
        raise ValueError(
            "method must be one of: None, 'moving_average', 'gaussian', 'savgol'"
        )


def current_density_from_enclosed_current(r, Iencl):
    """
    Recover J(r) from enclosed current Iencl(r):

        J(r) = (1 / (2*pi*r)) * dIencl/dr

    Parameters
    ----------
    r     : 1D array [cm]
    Iencl : 1D array [A] or [kA]

    Returns
    -------
    J : 1D array [A/cm^2] or [kA/cm^2], consistent with Iencl units
    """
    r = np.asarray(r, dtype=float)
    Iencl = np.asarray(Iencl, dtype=float)

    if r.ndim != 1 or Iencl.ndim != 1 or len(r) != len(Iencl):
        raise ValueError("r and Iencl must be 1D arrays of the same length")

    dIdr = np.gradient(Iencl, r)
    J = np.empty_like(Iencl)

    mask = r > 0
    J[mask] = dIdr[mask] / (2.0 * np.pi * r[mask])

    # Estimate axis value from first off-axis point
    if np.any(~mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError("Need at least one point with r > 0")
        i1 = idx[0]
        J[~mask] = Iencl[i1] / (np.pi * r[i1] ** 2)

    return J