from .structure import StructureRZ, assign_electrodes_by_extent
from matplotlib.collections import LineCollection
import numpy as np

def draw_structure_rz(ax, struct, color="white", linewidth=1.5):
    """
    Draw structure segments using a fast LineCollection.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on
    struct : StructureRZ
        Structure object produced by StructureRZ.from_p4structure()
    color : str
        Line color
    linewidth : float
        Line width
    """

    if not hasattr(struct, "segments"):
        raise TypeError(
            "draw_structure_rz expects a StructureRZ object. "
            "Call StructureRZ.from_p4structure() first."
        )

    segments = []

    for seg in struct.segments:
        segments.append(
            [(seg.z0, seg.r0), (seg.z1, seg.r1)]
        )

    lc = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        zorder=100
    )

    ax.add_collection(lc)

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
    bodies = struct.find_connected_bodies()
    anode_id, cathode_ids = assign_electrodes_by_extent(struct, polarity=polarity)

    for i, body in enumerate(bodies):
        color = anode_color if i == anode_id else cathode_color
        draw_body(ax, struct, body, color=color, linewidth=linewidth)

    return ax