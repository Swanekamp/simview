from .structure import StructureRZ, assign_electrodes_by_extent
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np

def draw_structure_bodies_debug(ax, struct, linewidth=2.0):
    bodies = struct.find_connected_bodies()

    debug_colors = [
        "red", "blue", "green", "orange", "cyan",
        "magenta", "yellow", "white"
    ]

    for ibody, body in enumerate(bodies):
        lines = []
        rsum = 0.0
        zsum = 0.0
        npts = 0

        for i in body:
            seg = struct.segments[i]
            lines.append([(seg.z0, seg.r0), (seg.z1, seg.r1)])

            rsum += seg.r0 + seg.r1
            zsum += seg.z0 + seg.z1
            npts += 2

        color = debug_colors[ibody % len(debug_colors)]

        ax.add_collection(LineCollection(
            lines,
            colors=color,
            linewidths=linewidth,
            zorder=110
        ))

        # label body near centroid
        zc = zsum / npts
        rc = rsum / npts
        ax.text(zc, rc, str(ibody), color=color, fontsize=12, zorder=120)

    return ax

def draw_structure_rz(ax, struct, polarity="negative",
                      foil_mid=None, linewidth=1.5,
                      fill=True):

    # Use existing bodies if available
    if not hasattr(struct, "bodies"):
        bodies = struct.find_connected_bodies()
    else:
        bodies = struct.bodies

    # Determine electrodes based on geometry
    anode_id, cathode_ids = assign_electrodes_by_extent(struct, polarity=polarity)

    body_color = {}

    for ibody in range(len(bodies)):
        if ibody == anode_id:
            body_color[ibody] = "red"
        else:
            body_color[ibody] = "blue"

    anode_segments = []
    cathode_segments = []
    foil_segments = []

    for ibody, body in enumerate(bodies):

        # default if body not classified
        color = body_color.get(ibody, "blue")

        if fill:

            rvals = []
            zvals = []

            for i in body:
                seg = struct.segments[i]
                rvals.extend([seg.r0, seg.r1])
                zvals.extend([seg.z0, seg.z1])

            rmin = min(rvals)
            rmax = max(rvals)
            zmin = min(zvals)
            zmax = max(zvals)

            hatch = "\\\\" if color == "red" else "///"

            rect = Rectangle(
                (zmin, rmin),
                zmax - zmin,
                rmax - rmin,
                facecolor="none",
                edgecolor=color,
                hatch=hatch,
                linewidth=0.5,
                zorder=90
            )

            ax.add_patch(rect)

        for i in body:

            seg = struct.segments[i]
            line = [(seg.z0, seg.r0), (seg.z1, seg.r1)]
            # --- classify foil FIRST ---

            # detect foil interface
            if foil_mid is not None and (seg.mid == foil_mid or seg.nid == foil_mid):
                foil_segments.append(line)
                continue

            # --- then classify electrode ---
            if color == "red":
                anode_segments.append(line)
            else:
                cathode_segments.append(line)

    if anode_segments:
        ax.add_collection(LineCollection(
            anode_segments,
            colors="red",
            linewidths=linewidth,
            zorder=100
        ))

    if cathode_segments:
        ax.add_collection(LineCollection(
            cathode_segments,
            colors="blue",
            linewidths=linewidth,
            zorder=100
        ))

    if foil_segments:
        ax.add_collection(LineCollection(
            foil_segments,
            colors="gray",
            linewidths=linewidth,
            zorder=101
        ))

    return ax

def draw_body(ax, struct, body_indices, color="blue", linewidth=2.0):
    """
    Draw one connected body given a list of segment indices.
    """
    for j in body_indices:
        seg = struct.segments[j]
        ax.plot(
            [seg.z0, seg.z1],
            [seg.r0, seg.r1],
            color=color,
            linewidth=linewidth,
            zorder=110,
        )

    return ax


def draw_structure_rz_colored(ax, struct, polarity="negative", linewidth=1.5,
                              anode_color="red", cathode_color="blue"):

    if not hasattr(struct, "bodies"):
        struct.find_connected_bodies()

    bodies = struct.bodies
    anode_id, cathode_ids = assign_electrodes_by_extent(struct, polarity=polarity)

    for i, body in enumerate(bodies):
        color = anode_color if i == anode_id else cathode_color
        draw_body(ax, struct, body, color=color, linewidth=linewidth)

    return ax