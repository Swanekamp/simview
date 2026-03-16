from pathlib import Path
import matplotlib.pyplot as plt
from .contour_plot import contour_plot


def make_batch_contours(
    files,
    *,
    load_fn,
    compute_fn,
    outdir,
    plot_name,
    fmt="png",
    dpi=250,
    **plot_kwargs
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for f in sorted(files):
        print("Processing", f)

        data = load_fn(f)
        r, z, F = compute_fn(data)

        t = getattr(data, "time", 0)

        fig, ax = contour_plot(
            z,
            r,
            F,
            title=plot_kwargs.get("title", "").format(time=t),
            cbar_label=plot_kwargs.get("cbar_label"),
            vmin=plot_kwargs.get("vmin"),
            vmax=plot_kwargs.get("vmax"),
            levels=plot_kwargs.get("levels", 100),
            line_levels=plot_kwargs.get("line_levels"),
            cmap=plot_kwargs.get("cmap", "viridis"),
            figsize=plot_kwargs.get("figsize", (7,3)),
        )

        fname = outdir / f"{plot_name}_{int(round(t)):03d}ns.{fmt}"
        fig.savefig(fname, dpi=dpi)

        plt.close(fig)

    print("Done.")