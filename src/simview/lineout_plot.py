# simview/lineout_plot.py

import numpy as np
import matplotlib.pyplot as plt


def lineout_plot(
    z,
    r,
    F,
    *,
    axis="z",
    value=None,
    index=None,
    title=None,
    xlabel=None,
    ylabel=None,
    label=None,
    xlim=None,
    ylim=None,
    logy=False,
    figsize=(8, 8),
    lw=2.0,
    time=None,
    time_units="ns",
    append_time_to_title=True,
    smoothing=None,
    smoothing_kwargs=None,
):
    """
    Extract and plot a 1D lineout from a 2D field F(z, r).
    """

    # --- Shape handling ---
    if F.shape == (len(z), len(r)):
        F_plot = F.T
    elif F.shape == (len(r), len(z)):
        F_plot = F
    else:
        raise ValueError(
            f"Field array shape {F.shape} does not match "
            f"(len(z), len(r)) = {(len(z), len(r))}"
        )

    if smoothing_kwargs is None:
        smoothing_kwargs = {}

    # --- Extract lineout ---
    if axis == "z":
        if value is not None:
            index = int(np.argmin(np.abs(r - value)))
        elif index is None:
            raise ValueError("For axis='z', provide either value or index.")

        x = z
        y = F_plot[index, :]
        slice_value = r[index]

        if xlabel is None:
            xlabel = "z (cm)"
        if ylabel is None:
            ylabel = "Value"
        if title is None:
            title = f"Lineout at r = {slice_value:.3g} cm"

    elif axis == "r":
        if value is not None:
            index = int(np.argmin(np.abs(z - value)))
        elif index is None:
            raise ValueError("For axis='r', provide either value or index.")

        x = r
        y = F_plot[:, index]
        slice_value = z[index]

        if xlabel is None:
            xlabel = "r (cm)"
        if ylabel is None:
            ylabel = "Value"
        if title is None:
            title = f"Lineout at z = {slice_value:.3g} cm"

    else:
        raise ValueError("axis must be 'z' or 'r'")

    # --- Optional smoothing ---
    if smoothing is not None:

        if smoothing == "moving_average":
            window = int(smoothing_kwargs.get("window", 5))
            kernel = np.ones(window) / window
            y = np.convolve(y, kernel, mode="same")

        elif smoothing == "gaussian":
            from scipy.ndimage import gaussian_filter1d

            sigma = float(smoothing_kwargs.get("sigma", 1.0))
            y = gaussian_filter1d(y, sigma=sigma)

        elif smoothing == "savgol":
            from scipy.signal import savgol_filter

            window_length = int(smoothing_kwargs.get("window_length", 7))
            polyorder = int(smoothing_kwargs.get("polyorder", 2))

            if window_length % 2 == 0:
                window_length += 1

            y = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        else:
            raise ValueError(
                "smoothing must be one of: None, 'moving_average', 'gaussian', 'savgol'"
            )

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Force square plotting region
    ax.set_box_aspect(1)

    # --- Plot ---
    if logy:
        y_plot = np.where(y > 0, y, np.nan)
        ax.semilogy(x, y_plot, lw=lw, label=label)
    else:
        ax.plot(x, y, lw=lw, label=label)

    # --- Title handling ---
    if append_time_to_title and time is not None:
        title = (
            f"{title}\n"
            f"t = {time:.1f} {time_units}"
        )

    ax.set_title(title, fontsize=22)

    # --- Labels ---
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    # --- Ticks + grid ---
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)

    # --- Limits ---
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif not logy:
        yfinite = y[np.isfinite(y)]
        if len(yfinite) > 0:
            ymin = np.min(yfinite)
            ymax = np.max(yfinite)
            yrange = ymax - ymin
            pad = 0.06 * yrange if yrange > 0 else 1.0
            ax.set_ylim(ymin - pad, ymax + pad)

    # --- Legend ---
    if label is not None:
        ax.legend(fontsize=12)

    return fig, ax


# --- Optional helper for saving ---
def save_lineout(fig, filename, dpi=150):
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt


def plot_1d_line(
    x,
    y,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    label=None,
    xlim=None,
    ylim=None,
    logy=False,
    figsize=(8, 8),
    lw=2.0,
    time=None,
    time_units="ns",
    append_time_to_title=True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_box_aspect(1)

    if logy:
        y = np.where(np.asarray(y) > 0, y, np.nan)
        ax.semilogy(x, y, lw=lw, label=label)
    else:
        ax.plot(x, y, lw=lw, label=label)

    if append_time_to_title and time is not None:
        if title is None:
            title = f"t = {time:.1f} {time_units}"
        else:
            title = f"{title}\nt = {time:.1f} {time_units}"

    if title is not None:
        ax.set_title(title, fontsize=22)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)

    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if label is not None:
        ax.legend(fontsize=12)

    return fig, ax