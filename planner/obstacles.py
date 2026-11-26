"""Convex polygon obstacles and collision checking.

A convex polygon is stored by its corners in counter-clockwise order.
For each face f we also store the outward normal n_f and offset a_f, so that
    point p is OUTSIDE face f   <=>   n_f . p >= a_f
(needed later for the big-M constraints in the MPC).
"""
import numpy as np


class Polygon:
    def __init__(self, vertices):
        V = np.asarray(vertices, dtype=float)
        # make sure the corners are counter-clockwise (positive signed area)
        area = 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
        if area < 0:
            V = V[::-1]
        self.vertices = V
        edges = np.roll(V, -1, axis=0) - V                 # edge f goes from corner f to f+1
        n = np.column_stack([edges[:, 1], -edges[:, 0]])   # rotate edge by -90 deg: points outward
        self.normals = n / np.linalg.norm(n, axis=1, keepdims=True)
        self.offsets = np.sum(self.normals * V, axis=1)    # a_f = n_f . (a corner on face f)

    def contains(self, p):
        """True if point p is inside the polygon (inside every face)."""
        return bool(np.all(self.normals @ p < self.offsets))


def rectangle(cx, cy, w, h):
    """Axis-aligned rectangle with centre (cx, cy), width w and height h."""
    return Polygon([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def polygons_intersect(A, B):
    """Separating axis test for two convex polygons: they do NOT overlap
    if some face normal of A or B separates their corners."""
    for n in np.vstack([A.normals, B.normals]):
        pa, pb = A.vertices @ n, B.vertices @ n
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True
