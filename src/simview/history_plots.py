import matplotlib.pyplot as plt

def plot_history(hist, y, x="time", ax=None, label=None,
                 xlabel=None, ylabel=None, title=None,
                 xlim=None, ylim=None, grid=True, legend=True, 
                 xscale=1.0, yscale=1.0, xunit=None, yunit=None, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    xdata = hist[x] * xscale
    if xunit is not None:
        xlabel = f"{xlabel} [{xunit}]"
        
    ydata = hist[y] * yscale
    if yunit is not None:
        ylabel = f"{ylabel} [{yunit}]"

    if label is None and isinstance(y, str):
        try:
            label = hist.get_short_label(y)
        except KeyError:
            label = y

    ax.plot(xdata, ydata, label=label, **plot_kwargs)

    if xlabel is None:
        if isinstance(x, str):
            xlabel = f"{hist.get_label(x)} [{hist.get_unit(x)}]"
        else:
            xlabel = "x"

    if ylabel is None:
        if isinstance(y, str):
            ylabel = f"{hist.get_short_label(y)} [{hist.get_unit(y)}]"
        else:
            ylabel = "y"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=0.3)
    if legend and label is not None:
        ax.legend()

    return fig, ax


def plot_histories(hist, ys, x="time", ax=None, labels=None,
                   xlabel=None, ylabel=None, title=None,
                   xlim=None, ylim=None, grid=True, legend=True, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if labels is None:
        labels = [None] * len(ys)

    for y, lbl in zip(ys, labels):
        label = lbl
        if label is None and isinstance(y, str):
            try:
                label = hist.get_short_label(y)
            except KeyError:
                label = y
        ax.plot(hist[x], hist[y], label=label, **plot_kwargs)

    if xlabel is None:
        if isinstance(x, str):
            xlabel = f"{hist.get_label(x)} [{hist.get_unit(x)}]"
        else:
            xlabel = "x"

    if ylabel is None:
        ylabel = "value"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()

    return fig, ax