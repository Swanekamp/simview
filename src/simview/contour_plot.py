# simview/contour.py

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.ticker import NullLocator, NullFormatter, LogLocator, LogFormatterMathtext


def _is_log_spaced(x, rtol=1e-3):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3:
        return False
    if np.any(x <= 0):
        return False
    lx = np.log10(x)
    d = np.diff(lx)
    return np.allclose(d, d[0], rtol=rtol, atol=0)


def contour_plot(
    z,
    r,
    F,
    *,
    title=None,
    cbar_label=None,
    vmin=None,
    vmax=None,
    levels=100,
    line_levels=None,
    cbar_ticks=None,
    cmap="viridis",
    figsize=(8, 6),
    logscale=False,
    label_contours=False,
):
    if F.shape == (len(z), len(r)):
        F_plot = F.T
    elif F.shape == (len(r), len(z)):
        F_plot = F
    else:
        raise ValueError(
            f"Field array shape {F.shape} does not match "
            f"(len(z), len(r)) = {(len(z), len(r))}"
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=18)

    if logscale:
        # Log plots must not contain non-positive values
        F_plot = np.where(F_plot > 0, F_plot, np.nan)

        if vmin is None:
            vmin = np.nanmin(F_plot)
        if vmax is None:
            vmax = np.nanmax(F_plot)

        # Continuous log-colored field
        cf = ax.pcolormesh(
            z,
            r,
            F_plot,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            shading="auto",
        )

        if line_levels is not None:
            cs = ax.contour(
                z,
                r,
                F_plot,
                levels=line_levels,
                colors="k",
                linewidths=1.0,
            )

            if label_contours:
                def fmt(x):
                    p = np.log10(x)
                    if np.isclose(p, round(p)):
                        return rf"$10^{{{int(round(p))}}}$"
                    return f"{x:.0f}"

                ax.clabel(cs, cs.levels, inline=True, fontsize=10, fmt=fmt)

    else:
        cf = ax.contourf(
            z,
            r,
            F_plot,
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        if line_levels is not None:
            cs = ax.contour(
                z,
                r,
                F_plot,
                levels=line_levels,
                colors="k",
                linewidths=1.0,
            )

            if label_contours:
                ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8)

    ax.set_xlabel("z (cm)", fontsize=20)
    ax.set_ylabel("r (cm)", fontsize=20)

    if title:
        ax.set_title(title, fontsize=24)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.tick_params(labelsize=12)

    if logscale:
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        # Let the colorbar use true logarithmic tick placement
        cbar.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        cbar.ax.yaxis.set_minor_locator(NullLocator())
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
        cbar.ax.yaxis.offsetText.set_visible(False)

    elif cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    if cbar_label:
        cbar.set_label(cbar_label, fontsize=14)

    return fig, ax