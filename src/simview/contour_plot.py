# simview/contour.py

import matplotlib.pyplot as plt

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
    figsize=(7, 3),
):
    
    if F.shape == (len(z), len(r)):
        F_plot = F.T
    elif F.shape == (len(r), len(z)):
        F_plot = F
    else:
        raise ValueError(
            f"Field array shape {F.shape} does not match "
            f"(len(z),len(r)) = {(len(z),len(r))}"
        )
    
    if figsize=="auto":
        # auto-adjust aspect ratio based on data range
        r_range = r.max() - r.min()
        z_range = z.max() - z.min()
        aspect = r_range / z_range if z_range > 0 else 1
        figsize = (13 , 13* aspect)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.tick_params(labelsize=18)

    cf = ax.contourf(z, r, F_plot,
                     levels=levels,
                     vmin=vmin,
                     vmax=vmax,
                     cmap=cmap)

    if line_levels is not None:
        cs = ax.contour(z, r, F_plot, levels=line_levels, colors="k", linewidths=0.6)
        # Label only every other line if desired:
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8)

    ax.set_xlabel("z (cm)",fontsize=20)
    ax.set_ylabel("r (cm)",fontsize=20)

    if title:
        ax.set_title(title, fontsize=24)


    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.tick_params(labelsize=12)

    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    if cbar_label:
        cbar.set_label(cbar_label, fontsize=14)

    return fig, ax