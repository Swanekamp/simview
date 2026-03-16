import numpy as np

def smooth_field(F, method=None, axis=None, **kwargs):
    """
    Reusable smoothing helper.

    Parameters
    ----------
    F : ndarray
        Input array.
    method : {None, "gaussian", "box", "three_point", "five_point", "savgol"}
        Smoothing method.
    axis : int or None
        Axis to smooth along. If None, use all axes for methods that support it.
    kwargs : dict
        Extra method-specific parameters.

    Returns
    -------
    ndarray
        Smoothed array.
    """
    if method is None:
        return F

    if method == "gaussian":
        from scipy.ndimage import gaussian_filter, gaussian_filter1d
        sigma = kwargs.get("sigma", 1.0)
        if axis is None:
            return gaussian_filter(F, sigma=sigma)
        return gaussian_filter1d(F, sigma=sigma, axis=axis)

    if method == "box":
        from scipy.ndimage import uniform_filter, uniform_filter1d
        size = kwargs.get("size", 3)
        if axis is None:
            return uniform_filter(F, size=size)
        return uniform_filter1d(F, size=size, axis=axis)

    if method == "three_point":
        if axis is None:
            raise ValueError("three_point requires axis")
        G = np.array(F, copy=True)
        slc_mid = [slice(None)] * F.ndim
        slc_lo  = [slice(None)] * F.ndim
        slc_hi  = [slice(None)] * F.ndim
        slc_mid[axis] = slice(1, -1)
        slc_lo[axis]  = slice(0, -2)
        slc_hi[axis]  = slice(2, None)
        G[tuple(slc_mid)] = (
            0.25 * F[tuple(slc_lo)] +
            0.50 * F[tuple(slc_mid)] +
            0.25 * F[tuple(slc_hi)]
        )
        return G

    if method == "five_point":
        if axis is None:
            raise ValueError("five_point requires axis")
        G = np.array(F, copy=True)
        s0 = [slice(None)] * F.ndim
        s1 = [slice(None)] * F.ndim
        s2 = [slice(None)] * F.ndim
        s3 = [slice(None)] * F.ndim
        s4 = [slice(None)] * F.ndim
        sm = [slice(None)] * F.ndim
        s0[axis] = slice(0, -4)
        s1[axis] = slice(1, -3)
        sm[axis] = slice(2, -2)
        s3[axis] = slice(3, -1)
        s4[axis] = slice(4, None)
        G[tuple(sm)] = (
            0.0625 * F[tuple(s0)] +
            0.25   * F[tuple(s1)] +
            0.375  * F[tuple(sm)] +
            0.25   * F[tuple(s3)] +
            0.0625 * F[tuple(s4)]
        )
        return G

    if method == "savgol":
        from scipy.signal import savgol_filter
        if axis is None:
            raise ValueError("savgol requires axis")
        window_length = kwargs.get("window_length", 5)
        polyorder = kwargs.get("polyorder", 2)
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(F, window_length=window_length, polyorder=polyorder, axis=axis)

    raise ValueError(f"Unknown smoothing method: {method}")