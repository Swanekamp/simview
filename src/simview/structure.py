from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SegmentRZ:
    r0: float
    z0: float
    r1: float
    z1: float
    mty: int
    mid: int
    nty: int
    nid: int


@dataclass
class StructureRZ:
    segments: list[SegmentRZ]

    @classmethod
    def from_p4structure(cls, STRUCT):

        segments = [
            SegmentRZ(r0, z0, r1, z1, mty, mid, nty, nid)
            for r0, z0, r1, z1, mty, mid, nty, nid in zip(
                STRUCT.xa,
                STRUCT.za,
                STRUCT.xb,
                STRUCT.zb,
                STRUCT.mty,
                STRUCT.mid,
                STRUCT.nty,
                STRUCT.nid,
            )
        ]

        obj = cls(segments)
        obj.find_connected_bodies()

        return obj
    
    def group_bodies_by_radius(self, tol=0.5):

        if not hasattr(self, "bodies"):
            self.find_connected_bodies()

        body_r = []

        for body in self.bodies:

            rvals = []

            for seg_id in body:
                seg = self.segments[seg_id]
                rvals.append(seg.r0)
                rvals.append(seg.r1)

            rmin = min(rvals)
            rmax = max(rvals)

            rc = 0.5 * (rmin + rmax)   # conductor center radius

            body_r.append(rc)

        order = sorted(range(len(body_r)), key=lambda i: body_r[i], reverse=True)

        groups = []

        for i in order:

            if not groups:
                groups.append([i])
                continue

            r0 = body_r[i]
            rref = body_r[groups[-1][0]]

            if abs(r0 - rref) < tol:
                groups[-1].append(i)
            else:
                groups.append([i])

        return groups
    
    def assign_electrodes(self, polarity="negative"):

        groups = self.group_bodies_by_radius()

        colors = {}

        for i, g in enumerate(groups):

            if polarity == "negative":
                electrode = "anode" if i % 2 == 0 else "cathode"
            else:
                electrode = "cathode" if i % 2 == 0 else "anode"

            for body in g:
                colors[body] = electrode

        return colors

    def find_connected_bodies(self, tol=1e-6):

        def key(r, z):
            return (round(r / tol), round(z / tol))

        # FIX: detect conductor on either side
        conductor_ids = [
            i for i, seg in enumerate(self.segments)
            if seg.mty == 1 or seg.nty == 1
        ]

        point_to_segments = {}

        for i in conductor_ids:

            seg = self.segments[i]

            for p in [key(seg.r0, seg.z0), key(seg.r1, seg.z1)]:
                point_to_segments.setdefault(p, []).append(i)

        visited = set()
        bodies = []

        for i in conductor_ids:

            if i in visited:
                continue

            stack = [i]
            body = []

            while stack:

                j = stack.pop()

                if j in visited:
                    continue

                visited.add(j)
                body.append(j)

                seg = self.segments[j]

                for p in [key(seg.r0, seg.z0), key(seg.r1, seg.z1)]:
                    for k in point_to_segments[p]:

                        if k not in visited:
                            stack.append(k)

            bodies.append(body)

        self.bodies = bodies

        return bodies


def assign_electrodes_by_extent(struct, polarity="negative"):
    """
    Returns (anode_id, cathode_ids) where IDs refer to connected bodies.
    """
    bodies = struct.find_connected_bodies()

    extents = []
    for i, body in enumerate(bodies):
        rmax = max(struct.segments[j].r0 for j in body)
        rmax = max(rmax, max(struct.segments[j].r1 for j in body))
        extents.append((rmax, i))

    anode_id = max(extents)[1]
    cathode_ids = [i for i in range(len(bodies)) if i != anode_id]

    if polarity == "positive" and cathode_ids:
        anode_id, cathode_ids = cathode_ids[0], [anode_id]

    return anode_id, cathode_ids