#%%
from pathlib import Path
import numpy as np

from simview.contour_plot import contour_plot
from simview.make_gif import make_gif
from simview.utils import _sorted_flds_files

# These are the Chicago P4 readers
from p4reader import P4Reader, P4Structure
from p4reader.inspect import print_available_p4_list, inspect_p4_files
from simview.draw_struct import draw_structure_rz
from simview.structure import StructureRZ
from simview.filters import smooth_field
import matplotlib.pyplot as plt
# ----------------------------
# File discovery
# ----------------------------

def sorted_p4_files(run_dir, pattern):
    return _sorted_flds_files(run_dir, pattern=pattern)


# ----------------------------
# Loaders
# ----------------------------

def load_flds(path):
    """
    Replace with your actual FLDS reader.
    Must return an object with:
      - time
      - r / x coordinate array
      - z coordinate array
      - field arrays
    """
    read_flds = P4Reader(path)
    return read_flds   # <-- adapt to your actual reader


def load_sclr(path):
    """
    Replace with your actual SCLR reader.
    """
    read_sclr = P4Reader(path)  # <-- adapt to your actual reader
    return read_sclr   # <-- adapt to your actual reader


# ----------------------------
# Coordinate helper
# ----------------------------

def get_rz(obj):
    """
    Return (r, z) from a CHICAGO object.

    Adapt this to whatever your p4 reader uses.
    Common possibilities:
      obj.x, obj.z
      obj.r, obj.z
    """
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

    # Adapt field name as needed:
    # Er = FLDS.Er
    # or Er = FLDS.Ex if CHICAGO stores radial as x
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
    from scipy.ndimage import gaussian_filter1d
    r, z = get_rz(FLDS)

    if hasattr(FLDS, field_name):
        F = getattr(FLDS, field_name)
    else:
        raise AttributeError(f"Could not find {field_name} in FLDS object")

    if smoothing_method is not None:
        if smoothing_method=='gaussian':
            sigma = kwargs.get("sigma", 1.0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=1)
        elif smoothing_method=='box':
            size = kwargs.get("size", 3)
            F = smooth_field(F, method=smoothing_method, size=size, axis=0)
            F = smooth_field(F, method=smoothing_method, size=size, axis=1)
        elif smoothing_method in ['three_point', 'five_point']:
            F = smooth_field(F, method=smoothing_method, axis=0)
            F = smooth_field(F, method=smoothing_method, axis=1)

    # convert current density to enclosed current
    Iencl = cumulative_trapezoid(2.0*np.pi*F*r, r, initial=0, axis=1) 
    return r, z, Iencl*scale # scale again to convert A to kA

def vector_mag(FLDS, field_name="J", smoothing_method=None, scale=1, kwargs=None):
    r, z = get_rz(FLDS)
    if hasattr(FLDS, field_name):
        V = getattr(FLDS, field_name)
    else:
        raise AttributeError("Could not find FLDS object")
    # Adapt names as needed
    Vx = V[:, :, 0]  # radial component
    Vy = V[:, :, 1]  # axial component

    F = np.sqrt(Vx**2 + Vy**2)
    if smoothing_method is not None:
        if smoothing_method=='gaussian':
            sigma = kwargs.get("sigma", 1.0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=0)
            F = smooth_field(F, method=smoothing_method, sigma=sigma, axis=1)
        elif smoothing_method=='box':
            size = kwargs.get("size", 3)
            F = smooth_field(F, method=smoothing_method, size=size, axis=0)
            F = smooth_field(F, method=smoothing_method, size=size, axis=1)
        elif smoothing_method in ['three_point', 'five_point']:
            F = smooth_field(F, method=smoothing_method, axis=0)
            F = smooth_field(F, method=smoothing_method, axis=1)
    return r, z, F * scale 

def compute_rBtheta(SCLR,scale=1e-3):
    r, z = get_rz(SCLR)

    # Adapt field name as needed
    # Could be SCLR.rBtheta, SCLR.rbtheta, or fetched by variable name
    if hasattr(SCLR, "rBtheta"):
        F = SCLR.rBtheta
    elif hasattr(SCLR, "rbtheta"):
        F = SCLR.rbtheta
    else:
        raise AttributeError("Could not find rBtheta in SCLR object")

    return r, z, F*scale # scale = 1e-3 scale Chicago rBtheta values from A to kA

def compute_named_field(obj, field_name, scale=1.0):
    # General helper to compute any field by name, with optional scaling. Adapts to whatever coordinate names the object uses.
    r, z = get_rz(obj)
    if not hasattr(obj, field_name):
        raise AttributeError(f"{field_name} not found")
    return r, z, getattr(obj, field_name) * scale

# ----------------------------
# Main plotting helper
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
    line_levels=None,
    levels=None,
    cbar_ticks=None,
):

    movie_dir = Path(run_dir) / "simview" / product_name
    movie_dir.mkdir(parents=True, exist_ok=True)

    frame = data
    r, z, F = compute_fn(frame)
    if vmin is None or vmax is None:
        vmin = np.nanmin(F)
        vmax = np.nanmax(F)

    # Handle constant fields (e.g. Jmag=0 early in the pulse)
    if np.isclose(vmin, vmax):
        eps = max(abs(vmin), 1.0) * 1e-6
        vmin -= eps
        vmax += eps

    if cbar_ticks is None:
        cbar_ticks = np.linspace(vmin, vmax, 9)

    nlevels = 200
    levels = np.linspace(vmin, vmax, nlevels)
    # contour settings
    if line_levels is None:
        line_levels=cbar_ticks
    
    t = getattr(frame, "time", 0)

    fig, ax = contour_plot(
        z,
        r,
        F,
        title = title.format(time=t),
        cbar_label=cbar_label,
        levels=levels,
        line_levels=line_levels,
        cbar_ticks=cbar_ticks,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if struct is not None:
        draw_structure_rz(ax, struct, linewidth=1.5, foil_mid=1, fill=False)

    fname = movie_dir / f"{product_name}_{int(round(t)):04d}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
#
# ---------------------------- Vector and scalar specifications
vector_specs = {

    "Er": dict(
        compute_fn=compute_Er,
        cmap="RdBu_r",
        cbar_label="Er (kV/cm)",
        vmin=-1500,
        vmax=1500,
        cbar_ticks=np.arange(-1500, 1501, 500),
        line_levels=np.arange(-1500,1501,250)
    ),

    "Ez": dict(
        compute_fn=compute_Ez,
        cmap="plasma",
        cbar_label="Ez (kV/cm)",
        vmin=-2500,
        vmax=500,
        line_levels=np.arange(-2500,251,250),
        cbar_ticks=np.arange(-2500, 501, 500),
    ),
    "By": dict(
        cmap="plasma",
        cbar_label="By (A/cm)",
        vmin = -20,
        vmax =  20,
        line_levels=np.arange(-20, 20 + 1,2),
        cbar_ticks=np.arange(-20, 20 + 1, 5),
        symmetric_cbar=True
    ),
    "Jz": dict(
        cmap="plasma",
        cbar_label="Jz (A/cm$^2$)",
        vmin=-40,
        vmax=0,
        line_levels=np.arange(-40,1,5),
        cbar_ticks=np.arange(-40, 1, 5),
        symmetric_cbar=False
    ),

    "Jb_mag": dict(
        compute_fn=lambda obj: vector_mag(obj, field_name="J", smoothing_method=None, scale=1, kwargs={"sigma": 0.1}),
        cmap="plasma",
        cbar_label="|J${_b}$| (A/cm²)",
        vmin=0,
        vmax=40,
        line_levels=np.arange(0, 41, 5),
        cbar_ticks=np.arange(0, 41, 5),
        symmetric_cbar=False,
    ),

    "Jp_mag": dict(
        compute_fn=lambda obj: vector_mag(obj, field_name="Jplasma", smoothing_method='gaussian', scale=1e-3, kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label="|J${_p}$| (kA/cm²)",
        vmin=0,
        vmax=50,
        line_levels=np.arange(0, 51, 5),
        cbar_ticks=np.arange(0, 51, 10),
    ),

    "E_mag": dict(
        compute_fn=lambda obj: vector_mag(obj, field_name="E", smoothing_method='gaussian', scale=1e-3, kwargs={"sigma": 3}),
        cmap="plasma",
        cbar_label="|E| (kV/cm)",
        vmin=0,
        vmax=50,
        line_levels=np.arange(0, 51, 5),
        cbar_ticks=np.arange(0, 51, 10),
    ),

    "Ib_encl": dict(
        compute_fn=lambda obj: compute_Iencl(obj, field_name="Jz", smoothing_method='gaussian', kwargs={"sigma":3}),
        cmap="plasma",
        cbar_label=r"$I_{b,\mathrm{encl}}$ (kA)",
        vmin=-800,
        vmax=0,
        line_levels=np.arange(-650, 1, 50),
        cbar_ticks=np.arange(-800, 1, 100),
    ),
    "Ip_encl": dict(
        compute_fn=lambda obj: compute_Iencl(obj, field_name="Jplasmaz", smoothing_method='gaussian', kwargs={"sigma":3}),
        cmap="plasma",
        cbar_label=r"$I_{p,\mathrm{encl}}$ (kA)",
        vmin=-1800,
        vmax=800,
        line_levels=np.arange(-1800, 801, 200),
        cbar_ticks=np.arange(-1800, 801, 200),
    ),
}

scalar_specs = {

    "rBtheta": dict(
        compute_fn=compute_rBtheta,
        cmap="RdBu_r",
        cbar_label=r"$rB_\theta$ (kA)",
        vmin=-400,
        vmax=400,
        levels=np.linspace(-400,400,101),
        line_levels=np.arange(-400,401,50)
    ),
}
# ----------------------------
# Top-level driver
# ----------------------------
def run_all(run_dir, plot="interactive"):

    run_dir = Path(run_dir)
    # Read structure if available
    struct_file = run_dir / "struct.p4"
    struct = None
    if struct_file.exists():
        STRUCT = P4Structure(struct_file)
        struct = StructureRZ.from_p4structure(STRUCT)
    else:
        STRUCT = None
    #
    # -----------------------------------
    # Inspect available fields
    # -----------------------------------

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

    # -----------------------------------
    # Determine what to plot
    # -----------------------------------

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
                raise RuntimeError(
                    f"\nInvalid fields requested:\n{invalid}\n"
                )

            selected_vectors = [v for v in chosen if v in known_vectors]
            selected_scalars = [s for s in chosen if s in known_scalars]

    print("\nPlotting vectors:", selected_vectors)
    print("Plotting scalars:", selected_scalars)

    # -----------------------------------
    # Discover files
    # -----------------------------------

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

            print(f"Computing global limits for {v}...")

            vmin, vmax = get_global_limits(flds_files, load_flds, compute_fn)

            global_limits[v] = (vmin, vmax)
    # -----------------------------------
    # FLDS loop
    # -----------------------------------
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
                # --- ALWAYS start fresh ---
                vmin = spec.get("vmin")
                vmax = spec.get("vmax")

                symmetric = spec.get("symmetric", False)
                user_defined = (vmin is not None and vmax is not None)

                # --- If not user-defined, pull from global ---
                if not user_defined:
                    vmin, vmax = global_limits[v]

                # --- Now apply logic ---
                if symmetric:
                    vmax = max(abs(vmin), abs(vmax))
                    if not user_defined:
                        vmax = 100 * np.ceil(vmax / 100)
                    vmin = -vmax
                else:
                    if vmin >= 0:
                        # purely positive field
                        vmin = 0
                    elif vmax <= 0:
                        # purely negative field
                        vmax = 0
                    # else: mixed sign → leave as-is
                    if not user_defined:
                        vmax = 100 * np.ceil(vmax / 100)
                cbar_ticks  = spec.get("cbar_ticks")
                line_levels = spec.get("line_levels")

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
                    levels = np.linspace(vmin, vmax, 201),
                    line_levels=line_levels,
                    struct=struct,
                )

    # -----------------------------------
    # SCLR loop
    # -----------------------------------
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
                line_levels = spec.get("line_levels")

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
                    line_levels=line_levels,
                    xlabel='z (cm)',
                    ylabel='x (cm)'
                )

    # -----------------------------------
    # GIF creation
    # -----------------------------------

    for name in selected_vectors + selected_scalars:

        outdir = run_dir / "simview" / name

        make_gif(outdir, outdir / f"{name}.gif", fps=2)

if __name__ == "__main__":
    PATH = "/home/swane/Runs/Xcimer/KJC/2D/run9"
    run_all(PATH)
