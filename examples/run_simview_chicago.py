#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from simview.contour_plot import contour_plot
from simview.make_gif import make_gif
from simview.utils import _sorted_flds_files

# These are the Chicago P4 readers
from p4reader import P4Reader, P4Structure
try:
    from p4reader.inspect import inspect_p4_files
except (ImportError, ModuleNotFoundError):
    from p4reader import inspect_p4_files
from simview.draw_struct import draw_structure_rz
from simview.structure import StructureRZ
from simview.filters import smooth_field


# ----------------------------
# File discovery
# ----------------------------

def sorted_p4_files(run_dir, pattern):
    return _sorted_flds_files(run_dir, pattern=pattern)


# ----------------------------
# Loaders
# ----------------------------

def load_flds(path):
    read_flds = P4Reader(path)
    return read_flds


def load_sclr(path):
    read_sclr = P4Reader(path)
    return read_sclr


# ----------------------------
# Coordinate helper
# ----------------------------

def get_rz(obj):
    if hasattr(obj, "r") and hasattr(obj, "z"):
        return obj.r, obj.z
    if hasattr(obj, "x") and hasattr(obj, "z"):
        return obj.x, obj.z
    raise AttributeError("Could not find radial and axial coordinates on object")


def get_global_limits(files, load_fn, compute_fn):
    vmin = np.inf
    vmax = -np.inf

    for f in files:
        data = load_fn(f)
        r, z, F = compute_fn(data)
        vmin = min(vmin, np.nanmin(F))
        vmax = max(vmax, np.nanmax(F))

    return vmin, vmax


# ----------------------------
# Compute functions
# ----------------------------

def compute_Er(FLDS):
    r, z = get_rz(FLDS)

    if hasattr(FLDS, "Er"):
        F = FLDS.Er
    elif hasattr(FLDS, "Ex"):
        F = FLDS.Ex
    else:
        raise AttributeError("Could not find Er/Ex in FLDS object")

    return r, z, F


def compute_Ez(FLDS):
    r, z = get_rz(FLDS)

    if hasattr(FLDS, "Ez"):
        F = FLDS.Ez
    else:
        raise AttributeError("Could not find Ez in FLDS object")

    return r, z, F


def compute_Iencl(FLDS, scale=1e-3, field_name="Jz", smoothing_method=None, kwargs=None):
    from scipy.integrate import cumulative_trapezoid

    if kwargs is None:
        kwargs = {}

    r, z = get_rz(FLDS)

    if hasattr(FLDS, field_name):
        F = getattr(FLDS, field_name)
    else:
        raise AttributeError(f"Could not find {field_name} in FLDS object")

    if smoothing_method is not None:
        if smoothing_method == "gaussian":
            sigma = kwargs.get("sigma", 1.0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=1)
        elif smoothing_method == "box":
            size = kwargs.get("size", 3)
            F = smooth_field(F, method=smoothing_method, size=size, axis=0)
            F = smooth_field(F, method=smoothing_method, size=size, axis=1)
        elif smoothing_method in ["three_point", "five_point"]:
            F = smooth_field(F, method=smoothing_method, axis=0)
            F = smooth_field(F, method=smoothing_method, axis=1)

    Iencl = cumulative_trapezoid(2.0 * np.pi * F * r, r, initial=0, axis=1)
    return r, z, Iencl * scale


def compute_Jz_from_Iencl(
    obj,
    *,
    axis="r",
    value=19.8,
    index=None,
    smoothing_method="gaussian",
    smoothing_kwargs=None,
):
    from simview.lineout import (
        extract_lineout,
        smooth_1d,
        current_density_from_enclosed_current,
    )

    if smoothing_kwargs is None:
        smoothing_kwargs = {}

    r, z, Iencl2d = compute_Iencl(
        obj,
        field_name="Jz",
        smoothing_method="gaussian",
        kwargs={"sigma": 3},
        scale=1e-3,  # kA
    )

    x, I_line = extract_lineout(
        z, r, Iencl2d,
        axis=axis,
        value=value,
        index=index,
    )

    I_line = smooth_1d(
        I_line,
        method=smoothing_method,
        kwargs=smoothing_kwargs,
    )

    J_line = current_density_from_enclosed_current(x, I_line)
    return x, J_line


def vector_mag(FLDS, field_name="J", smoothing_method=None, scale=1.0, kwargs=None):
    if kwargs is None:
        kwargs = {}

    r, z = get_rz(FLDS)

    if hasattr(FLDS, field_name):
        V = getattr(FLDS, field_name)
    else:
        raise AttributeError(f"Could not find {field_name} in FLDS object")

    Vx = V[:, :, 0]
    Vy = V[:, :, 1]
    Vz = V[:, :, 2]

    F = np.sqrt(Vx**2 + Vy**2 + Vz**2)

    if smoothing_method is not None:
        if smoothing_method == "gaussian":
            sigma = kwargs.get("sigma", 1.0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=1)
        elif smoothing_method == "box":
            size = kwargs.get("size", 3)
            F = smooth_field(F, method=smoothing_method, size=size, axis=0)
            F = smooth_field(F, method=smoothing_method, size=size, axis=1)
        elif smoothing_method in ["three_point", "five_point"]:
            F = smooth_field(F, method=smoothing_method, axis=0)
            F = smooth_field(F, method=smoothing_method, axis=1)

    return r, z, F * scale


def compute_rBtheta(SCLR, scale=1e-3):
    r, z = get_rz(SCLR)

    if hasattr(SCLR, "rBtheta"):
        F = SCLR.rBtheta
    elif hasattr(SCLR, "rbtheta"):
        F = SCLR.rbtheta
    else:
        raise AttributeError("Could not find rBtheta in SCLR object")

    return r, z, F * scale


def compute_named_field(obj, field_name, scale=1.0):
    r, z = get_rz(obj)
    if not hasattr(obj, field_name):
        raise AttributeError(f"{field_name} not found")
    return r, z, getattr(obj, field_name) * scale


# ----------------------------
# Main contour plotting helper
# ----------------------------

def make_movie_set(
    data,
    run_dir,
    compute_fn,
    product_name,
    title="",
    cbar_label="",
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    struct=None,
    levels=None,
    line_levels=None,
    cbar_ticks=None,
    logscale=False,
    label_contours=False,
):
    movie_dir = Path(run_dir) / "simview" / "contours" / product_name
    movie_dir.mkdir(parents=True, exist_ok=True)

    frame = data
    r, z, F = compute_fn(frame)

    if vmin is None or vmax is None:
        vmin = np.nanmin(F)
        vmax = np.nanmax(F)

    if np.isclose(vmin, vmax):
        eps = max(abs(vmin), 1.0) * 1e-6
        vmin -= eps
        vmax += eps

    if cbar_ticks is None:
        if logscale:
            cbar_ticks = np.logspace(
                np.log10(vmin),
                np.log10(vmax),
                int(np.log10(vmax) - np.log10(vmin)) + 1,
            )
        else:
            cbar_ticks = np.linspace(vmin, vmax, 9)

    if levels is None:
        if logscale:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 200)
        else:
            levels = np.linspace(vmin, vmax, 200)

    if line_levels is None:
        line_levels = cbar_ticks

    t = getattr(frame, "time", 0.0)

    fig, ax = contour_plot(
        z,
        r,
        F,
        title=title.format(time=t),
        cbar_label=cbar_label,
        levels=levels,
        line_levels=line_levels,
        cbar_ticks=cbar_ticks,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        logscale=logscale,
        label_contours=label_contours,
    )

    if struct is not None:
        draw_structure_rz(ax, struct, linewidth=1.5, foil_mid=1, fill=False)

    fname = movie_dir / f"{product_name}_{int(round(t)):04d}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ----------------------------
# Specs
# ----------------------------

lineout_specs = {
    "Jz_from_Iencl": dict(
        compute_lineout_fn=lambda obj: compute_Jz_from_Iencl(
            obj,
            axis="r",
            value=19.8,
            smoothing_method="gaussian",
            smoothing_kwargs={"sigma": 3.0},
        ),
        xlabel="r (cm)",
        logy=False,
        ylabel=r"$J_z$ (A/cm$^2$)",
        title=r"Lineout of $J_z(r)$ at $z=19.8$ cm",
        ylim=[-3, 0.5],
    ),
}

vector_specs = {
    "Er": dict(
        compute_fn=compute_Er,
        cmap="RdBu_r",
        cbar_label="Er (kV/cm)",
        vmin=-1500,
        vmax=1500,
        cbar_ticks=np.arange(-1500, 1501, 500),
        line_levels=np.arange(-1500, 1501, 250),
    ),

    "Ez": dict(
        compute_fn=compute_Ez,
        cmap="plasma",
        cbar_label="Ez (kV/cm)",
        vmin=-2500,
        vmax=500,
        line_levels=np.arange(-2500, 251, 250),
        cbar_ticks=np.arange(-2500, 501, 500),
    ),

    "By": dict(
        cmap="plasma",
        cbar_label="By (A/cm)",
        vmin=-20,
        vmax=20,
        line_levels=np.arange(-20, 21, 2),
        cbar_ticks=np.arange(-20, 21, 5),
        symmetric=True,
    ),

    "Jz": dict(
        compute_fn=lambda obj: compute_named_field(obj, "Jz", scale=1e-3),
        cmap="plasma",
        cbar_label="Jz (A/cm$^2$)",
        vmin=-3,
        vmax=0,
        line_levels=np.arange(-3, 1, 0.5),
        cbar_ticks=np.arange(-3, 1, 0.5),
        symmetric=False,
        smoothing="gaussian",
        smoothing_kwargs={"sigma": 3.0},
        lineout=dict(
            axis="r",
            value=9.9,
            logy=False,
            ylabel=r"$J_z$ (kA/cm$^2$)",
            title=r"Lineout of $J_z(r)$ at various z locations",
            smoothing="gaussian",
            smoothing_kwargs={"sigma": 3.0},
            ylim=[-3, 0.5],
        ),
    ),

    "Jb_mag": dict(
        compute_fn=lambda obj: vector_mag(obj, field_name="J", smoothing_method="gaussian", scale=1e-3, kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label="|J${_b}$| (kA/cm²)",
        vmin=0,
        vmax=5,
        line_levels=[0, 1, 2, 3, 4, 5],
        cbar_ticks=[0, 1, 2, 3, 4, 5],
        symmetric=False,
    ),

    "Jp_mag": dict(
        compute_fn=lambda obj: vector_mag(obj, field_name="Jplasma", smoothing_method="gaussian", scale=1e-3, kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label="|J${_p}$| (kA/cm²)",
    ),

    "E_mag": dict(
        compute_fn=lambda obj: vector_mag(obj, field_name="E", smoothing_method="gaussian", scale=1, kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label="|E| (V/cm)",
        vmin=10,
        vmax=10000,
        line_levels=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        cbar_ticks=np.logspace(1, 4, 4),
        logscale=True,
        label_contours=True,
    ),

    "Ib_encl": dict(
        compute_fn=lambda obj: compute_Iencl(obj, field_name="Jz", smoothing_method="gaussian", kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label=r"$I_{b,\mathrm{encl}}$ (kA)",
        vmin=-800,
        vmax=0,
        levels=np.linspace(-800, 0, 201),
        line_levels=np.arange(-800, 0, 25),
        cbar_ticks=np.arange(-800, 0, 100),
#
#        vmin=-800,
#        vmax=0,
#        line_levels=np.arange(-800, 1, 50),
#        cbar_ticks=np.arange(-800, 1, 100),
        lineout=dict(
            axis="r",
            value=19.8,
            logy=False,
            ylabel=r"$I_{b,\mathrm{encl}}$ (kA)",
            title=r"Lineout of $I_{b,\mathrm{encl}}(r)$ at $z=19.8$ cm",
            smoothing="gaussian",
            smoothing_kwargs={"sigma": 3.0},
            ylim=[-800, 50],
        ),
    ),

    "Ip_encl": dict(
        compute_fn=lambda obj: compute_Iencl(obj, field_name="Jplasmaz", smoothing_method="gaussian", kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label=r"$I_{p,\mathrm{encl}}$ (kA)",
        vmin=0,
        vmax=800,
        line_levels=np.arange(0, 801, 50),
        cbar_ticks=np.arange(0, 801, 100),
    ),

    "Isp1_encl": dict(
        compute_fn=lambda obj: compute_Iencl(obj, field_name="Jsp1z", smoothing_method="gaussian", kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label=r"$I_{sp1,\mathrm{encl}}$ (kA)",
        vmin=0,
        vmax=800,
        line_levels=np.arange(0, 801, 50),
        cbar_ticks=np.arange(0, 801, 100),
    ),

    "Isp2_encl": dict(
        compute_fn=lambda obj: compute_Iencl(obj, field_name="Jsp2z", smoothing_method="gaussian", kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label=r"$I_{sp2,\mathrm{encl}}$ (kA)",
        vmin=0,
        vmax=800,
        line_levels=np.arange(0, 801, 50),
        cbar_ticks=np.arange(0, 801, 100),
    ),
}

scalar_specs = {
    "rBtheta": dict(
        compute_fn=compute_rBtheta,
        cmap="RdBu_r",
        cbar_label=r"$rB_\theta$ (kA)",
        vmin=-400,
        vmax=400,
        levels=np.linspace(-400, 400, 201),
        line_levels=np.arange(-400, 401, 50),
        label_contours=True,
        lineout=dict(
            axis="r",
            value=0.1,
            logy=False,
            ylabel=r"$I_{b,\mathrm{encl}}$ (kA)",
            title=r"Lineout of $I_{b,\mathrm{encl}}(r)$ at $z=0.1$ cm",
#            smoothing="gaussian",
#            smoothing_kwargs={"sigma": 3.0},
            ylim=[-400, 401],
        ),
    ),

    "Edens": dict(
        cmap="viridis",
        cbar_label=r"$n_e$ (cm$^{-3}$)",
        vmin=1e10,
        vmax=1e16,
        line_levels=np.logspace(10, 16, 7),
        cbar_ticks=np.logspace(10, 16, 7),
        logscale=True,
        label_contours=True,
    ),

    "RhoN2": dict(
        cmap="plasma",
        cbar_label=r"$n_e$ (cm$^{-3}$)",
        vmin=1e12,
        vmax=1e16,
        levels=100,
        line_levels=np.logspace(12, 16, 5),
        cbar_ticks=10 ** np.arange(12, 17, 1),
        logscale=True,
    ),

    "Gdens": dict(
        cmap="plasma",
        cbar_label=r"$n_{Gas}$ (cm$^{-3}$)",
        vmin=1e12,
        vmax=1e19,
        levels=200,
        line_levels=np.logspace(12, 19, 8),
        cbar_ticks=10 ** np.arange(12, 20, 1),
        logscale=True,
    ),
}


# ----------------------------
# Contours driver
# ----------------------------

def run_contours(run_dir, plot="interactive"):
    run_dir = Path(run_dir)

    struct_file = run_dir / "struct.p4"
    struct = None
    if struct_file.exists():
        STRUCT = P4Structure(struct_file)
        struct = StructureRZ.from_p4structure(STRUCT)

    fields = inspect_p4_files(run_dir)
    vectors = fields["vectors"]
    scalars = fields["scalars"]

    print("\nAvailable vector components (raw + computed):")
    all_vectors = sorted(set(vectors) | set(vector_specs.keys()))
    for v in all_vectors:
        tag = "(computed)" if v in vector_specs else ""
        print(f"   {v:15s} {tag}")

    print("\nAvailable scalar fields (sclr*.p4):")
    for s in scalars:
        print("   ", s)

    if plot == "interactive":
        selected = input(
            "\nEnter fields to plot (comma separated) or 'all': "
        ).strip()

        if selected.lower() == "all":
            selected_vectors = vectors
            selected_scalars = scalars
        else:
            chosen = [x.strip() for x in selected.split(",") if x.strip()]

            known_vectors = set(vectors) | set(vector_specs.keys())
            known_scalars = set(scalars) | set(scalar_specs.keys())

            invalid = [
                x for x in chosen
                if x not in known_vectors and x not in known_scalars
            ]
            if invalid:
                raise RuntimeError(f"\nInvalid fields requested:\n{invalid}\n")

            selected_vectors = [v for v in chosen if v in known_vectors]
            selected_scalars = [s for s in chosen if s in known_scalars]
    else:
        selected_vectors = all_vectors
        selected_scalars = list(set(scalars) | set(scalar_specs.keys()))

    print("\nPlotting vectors:", selected_vectors)
    print("Plotting scalars:", selected_scalars)

    flds_files = sorted_p4_files(run_dir, "flds*.p4")
    sclr_files = sorted_p4_files(run_dir, "sclr*.p4")

    global_limits = {}

    if selected_vectors:
        for v in selected_vectors:
            spec = vector_specs.get(v, {})
            compute_fn = spec.get(
                "compute_fn",
                lambda obj, name=v: compute_named_field(obj, name)
            )

            user_vmin = spec.get("vmin")
            user_vmax = spec.get("vmax")

            if user_vmin is not None and user_vmax is not None:
                print(f"Using user-defined limits for {v}: ({user_vmin}, {user_vmax})")
                global_limits[v] = (user_vmin, user_vmax)
            else:
                print(f"Computing global limits for {v}...")
                gvmin, gvmax = get_global_limits(flds_files, load_flds, compute_fn)
                global_limits[v] = (gvmin, gvmax)

    if selected_vectors:
        for f in flds_files:
            print("Processing", f)
            FLDS = load_flds(f)

            for v in selected_vectors:
                spec = vector_specs.get(v, {})

                compute_fn = spec.get(
                    "compute_fn",
                    lambda obj, name=v: compute_named_field(obj, name)
                )

                cmap = spec.get("cmap", "plasma")
                cbar_label = spec.get("cbar_label", v)
                vmin = spec.get("vmin")
                vmax = spec.get("vmax")
                symmetric = spec.get("symmetric", False)
                logscale = spec.get("logscale", False)
                label_contours = spec.get("label_contours", False)

                user_defined = (vmin is not None and vmax is not None)
                if not user_defined:
                    vmin, vmax = global_limits[v]

                if symmetric:
                    vmax = max(abs(vmin), abs(vmax))
                    if not user_defined:
                        vmax = 100 * np.ceil(vmax / 100)
                    vmin = -vmax
                else:
                    if not logscale:
                        if vmin >= 0:
                            vmin = 0
                        elif vmax <= 0:
                            vmax = 0
                    if not user_defined:
                        vmax = 100 * np.ceil(vmax / 100)

                cbar_ticks = spec.get("cbar_ticks")
                line_levels = spec.get("line_levels")
                levels = None if logscale else np.linspace(vmin, vmax, 201)

                make_movie_set(
                    data=FLDS,
                    run_dir=run_dir,
                    compute_fn=compute_fn,
                    product_name=v,
                    title=f"{v}  t = {{time:.1f}} ns",
                    cbar_label=cbar_label,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    cbar_ticks=cbar_ticks,
                    levels=levels,
                    line_levels=line_levels,
                    logscale=logscale,
                    label_contours=label_contours,
                    struct=struct,
                )

    if selected_scalars:
        for f in sclr_files:
            print("Processing", f)
            SCLR = load_sclr(f)

            for s in selected_scalars:
                spec = scalar_specs.get(s, {})

                compute_fn = spec.get(
                    "compute_fn",
                    lambda obj, name=s: compute_named_field(obj, name)
                )

                cmap = spec.get("cmap", "viridis")
                cbar_label = spec.get("cbar_label", s)
                vmin = spec.get("vmin")
                vmax = spec.get("vmax")
                levels = spec.get("levels")
                line_levels = spec.get("line_levels")
                cbar_ticks = spec.get("cbar_ticks")
                logscale = spec.get("logscale", False)
                label_contours = spec.get("label_contours", False)

                make_movie_set(
                    data=SCLR,
                    run_dir=run_dir,
                    compute_fn=compute_fn,
                    product_name=s,
                    title=f"{s}  t = {{time:.1f}} ns",
                    cbar_label=cbar_label,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    struct=struct,
                    levels=levels,
                    line_levels=line_levels,
                    cbar_ticks=cbar_ticks,
                    logscale=logscale,
                    label_contours=label_contours,
                )

    for name in selected_vectors + selected_scalars:
        outdir = run_dir / "simview" / "contours" / name
        if outdir.exists() and any(outdir.glob("*.png")):
            make_gif(outdir, outdir / f"{name}.gif", fps=2)


# ----------------------------
# Lineouts driver
# ----------------------------

def run_lineouts(run_dir, plot="interactive"):
    from simview.lineout_plot import lineout_plot, plot_1d_line

    run_dir = Path(run_dir)

    flds_files = sorted_p4_files(run_dir, "flds*.p4")
    sclr_files = sorted_p4_files(run_dir, "sclr*.p4")

    ordinary_vector_lineouts = {
        name: spec for name, spec in vector_specs.items()
        if "lineout" in spec
    }
    ordinary_scalar_lineouts = {
        name: spec for name, spec in scalar_specs.items()
        if "lineout" in spec
    }
    derived_lineouts = dict(lineout_specs)

    available = sorted(
        list(ordinary_vector_lineouts.keys())
        + list(ordinary_scalar_lineouts.keys())
        + list(derived_lineouts.keys())
    )

    print("\nAvailable lineouts:")
    for k in available:
        print("   ", k)

    if plot == "interactive":
        selected = input(
            "\nEnter lineouts to plot (comma separated) or 'all': "
        ).strip()

        if selected.lower() == "all":
            selected_lineouts = available
        else:
            selected_lineouts = [x.strip() for x in selected.split(",") if x.strip()]
    else:
        selected_lineouts = available

    print("Plotting lineouts:", selected_lineouts)

    selected_vector = [n for n in selected_lineouts if n in ordinary_vector_lineouts]
    selected_scalar = [n for n in selected_lineouts if n in ordinary_scalar_lineouts]
    selected_derived = [n for n in selected_lineouts if n in derived_lineouts]

    if selected_vector or selected_derived:
        for f in flds_files:
            print("Processing", f)
            FLDS = load_flds(f)
            t = getattr(FLDS, "time", 0.0)

            for name in selected_vector:
                spec = ordinary_vector_lineouts[name]
                lineout_spec = spec["lineout"]

                compute_fn = spec.get(
                    "compute_fn",
                    lambda obj, field_name=name: compute_named_field(obj, field_name)
                )

                r, z, F = compute_fn(FLDS)

                fig, ax = lineout_plot(
                    z,
                    r,
                    F,
                    axis=lineout_spec.get("axis", "z"),
                    value=lineout_spec.get("value"),
                    index=lineout_spec.get("index"),
                    ylabel=lineout_spec.get("ylabel", name),
                    title=f"{lineout_spec.get('title', name)}\nt = {t:.1f} ns",
                    logy=lineout_spec.get("logy", False),
                    ylim=lineout_spec.get("ylim"),
                    smoothing=lineout_spec.get("smoothing"),
                    smoothing_kwargs=lineout_spec.get("smoothing_kwargs"),
                )

                outdir = run_dir / "simview" / "lineouts" / name
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"{name}_{int(round(t)):04d}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)

            for name in selected_derived:
                spec = derived_lineouts[name]
                compute_lineout_fn = spec["compute_lineout_fn"]

                x, y = compute_lineout_fn(FLDS)

                fig, ax = plot_1d_line(
                    x,
                    y,
                    xlabel=spec.get("xlabel", "r (cm)"),
                    ylabel=spec.get("ylabel", name),
                    title=f"{spec.get('title', name)}\nt = {t:.1f} ns",
                    logy=spec.get("logy", False),
                    ylim=spec.get("ylim"),
                )

                outdir = run_dir / "simview" / "lineouts" / name
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"{name}_{int(round(t)):04d}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)

    if selected_scalar:
        for f in sclr_files:
            print("Processing", f)
            SCLR = load_sclr(f)
            t = getattr(SCLR, "time", 0.0)

            for name in selected_scalar:
                spec = ordinary_scalar_lineouts[name]
                lineout_spec = spec["lineout"]

                compute_fn = spec.get(
                    "compute_fn",
                    lambda obj, field_name=name: compute_named_field(obj, field_name)
                )

                r, z, F = compute_fn(SCLR)

                fig, ax = lineout_plot(
                    z,
                    r,
                    F,
                    axis=lineout_spec.get("axis", "z"),
                    value=lineout_spec.get("value"),
                    index=lineout_spec.get("index"),
                    ylabel=lineout_spec.get("ylabel", name),
                    title=f"{lineout_spec.get('title', name)}\nt = {t:.1f} ns",
                    logy=lineout_spec.get("logy", False),
                    ylim=lineout_spec.get("ylim"),
                    smoothing=lineout_spec.get("smoothing"),
                    smoothing_kwargs=lineout_spec.get("smoothing_kwargs"),
                )

                outdir = run_dir / "simview" / "lineouts" / name
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"{name}_{int(round(t)):04d}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    import sys

    mode = "both"
    PATH = "/home/swane/Runs/Syntek/Chicago/Fisica/run64"
    plot = "interactive"

    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        key = key.lower()

        if key == "mode":
            mode = value.lower()
        elif key == "path":
            PATH = value
        elif key == "plot":
            plot = value.lower()

    if mode == "contours":
        run_contours(PATH, plot=plot)
    elif mode == "lineouts":
        run_lineouts(PATH, plot=plot)
    elif mode == "both":
        run_contours(PATH, plot=plot)
        run_lineouts(PATH, plot=plot)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use: contours, lineouts, or both."
        )