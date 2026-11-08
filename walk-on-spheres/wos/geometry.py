"""2D boundary representation for Walk on Spheres / Walk on Stars.

A domain is described by closed polygonal loops. Every segment of a loop is
either Dirichlet (prescribed value u = g) or Neumann (zero flux, du/dn = 0).
Loops are oriented so the domain interior is always on the *left* of each
segment, which makes the inward normal of a segment (-dy, dx) / |d|.

The Scene stores Dirichlet and Neumann segments in separate arrays, each
with its own BVH so closest-point and ray queries are O(log n).
"""

import numpy as np

DIRICHLET = 0
NEUMANN = 1


def signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


class BVH:
    """Bounding volume hierarchy over 2D line segments.

    Built once in Python; stored as flat arrays so numba kernels can traverse
    it. Segments are reordered so every leaf covers a contiguous range.
    """

    LEAF_SIZE = 4

    def __init__(self, a, b):
        n = len(a)
        self.order = np.arange(n)
        lo, hi, left, right, start, count = [], [], [], [], [], []

        if n == 0:
            self.lo = np.zeros((0, 2))
            self.hi = np.zeros((0, 2))
            self.left = np.zeros(0, np.int64)
            self.right = np.zeros(0, np.int64)
            self.start = np.zeros(0, np.int64)
            self.count = np.zeros(0, np.int64)
            return

        seg_lo = np.minimum(a, b)
        seg_hi = np.maximum(a, b)
        centroid = 0.5 * (a + b)
        order = np.arange(n)

        # Iterative build: (node index, begin, end) into `order`.
        def new_node():
            lo.append(None); hi.append(None)
            left.append(-1); right.append(-1)
            start.append(0); count.append(0)
            return len(lo) - 1

        root = new_node()
        stack = [(root, 0, n)]
        while stack:
            node, s, e = stack.pop()
            idx = order[s:e]
            lo[node] = seg_lo[idx].min(axis=0)
            hi[node] = seg_hi[idx].max(axis=0)
            start[node], count[node] = s, e - s
            if e - s <= self.LEAF_SIZE:
                continue
            c = centroid[idx]
            axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
            mid = (e - s) // 2
            part = np.argpartition(c[:, axis], mid)
            order[s:e] = idx[part]
            l, r = new_node(), new_node()
            left[node], right[node] = l, r
            stack.append((l, s, s + mid))
            stack.append((r, s + mid, e))

        self.order = order
        self.lo = np.array(lo, dtype=np.float64)
        self.hi = np.array(hi, dtype=np.float64)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.start = np.array(start, dtype=np.int64)
        self.count = np.array(count, dtype=np.int64)

    def arrays(self):
        return self.lo, self.hi, self.left, self.right, self.start, self.count


class Scene:
    """A 2D domain made of closed loops of Dirichlet / Neumann segments."""

    def __init__(self):
        self._loops = []  # (points, kinds, values)
        self._built = False

    def add_loop(self, points, kinds=DIRICHLET, values=0.0, hole=False):
        """Add a closed polygon.

        points : (n, 2) vertices, without repeating the first vertex.
        kinds  : DIRICHLET / NEUMANN, scalar or per-segment array of length n
                 (segment i goes from vertex i to vertex i+1).
        values : Dirichlet boundary values, scalar, per-vertex array, or a
                 callable g(x, y) evaluated at the vertices. Values are
                 linearly interpolated along each segment.
        hole   : True if this loop bounds a hole (obstacle) inside the domain.
        """
        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)
        kinds = np.broadcast_to(np.asarray(kinds, dtype=np.int64), (n,)).copy()
        if callable(values):
            vals = np.asarray(values(pts[:, 0], pts[:, 1]), dtype=np.float64)
            vals = np.broadcast_to(vals, (n,)).copy()
        else:
            vals = np.broadcast_to(np.asarray(values, dtype=np.float64), (n,)).copy()

        # Outer boundaries counter-clockwise, holes clockwise => interior on the left.
        ccw = signed_area(pts) > 0
        if ccw == hole:
            pts = pts[::-1].copy()
            vals = vals[::-1].copy()
            # Segment i (v_i -> v_{i+1}) becomes segment n-2-i after reversal.
            kinds = np.roll(kinds[::-1], -1).copy()
        self._loops.append((pts, kinds, vals))
        self._built = False
        return self

    def build(self):
        if self._built:
            return self
        da, db, dga, dgb = [], [], [], []
        na, nb, nprev, nnext = [], [], [], []
        all_a, all_b = [], []
        for pts, kinds, vals in self._loops:
            n = len(pts)
            nxt = np.roll(np.arange(n), -1)
            base = len(na)
            # Map loop segment index -> Neumann array index.
            n_index = -np.ones(n, np.int64)
            k = 0
            for i in range(n):
                if kinds[i] == NEUMANN:
                    n_index[i] = base + k
                    k += 1
            for i in range(n):
                a, b = pts[i], pts[nxt[i]]
                all_a.append(a); all_b.append(b)
                if kinds[i] == DIRICHLET:
                    da.append(a); db.append(b)
                    dga.append(vals[i]); dgb.append(vals[nxt[i]])
                else:
                    na.append(a); nb.append(b)
                    nprev.append(n_index[(i - 1) % n])
                    nnext.append(n_index[nxt[i]])

        def arr(x, shape):
            return np.array(x, dtype=np.float64).reshape(shape)

        self.d_a, self.d_b = arr(da, (-1, 2)), arr(db, (-1, 2))
        self.d_ga, self.d_gb = arr(dga, (-1,)), arr(dgb, (-1,))
        self.n_a, self.n_b = arr(na, (-1, 2)), arr(nb, (-1, 2))
        self.n_prev = np.array(nprev, dtype=np.int64)
        self.n_next = np.array(nnext, dtype=np.int64)
        self.all_a, self.all_b = arr(all_a, (-1, 2)), arr(all_b, (-1, 2))

        if len(self.d_a) == 0:
            raise ValueError("Scene needs at least one Dirichlet segment")

        # Build BVHs and reorder segment data to BVH order.
        self.d_bvh = BVH(self.d_a, self.d_b)
        o = self.d_bvh.order
        self.d_a, self.d_b = self.d_a[o], self.d_b[o]
        self.d_ga, self.d_gb = self.d_ga[o], self.d_gb[o]

        self.n_bvh = BVH(self.n_a, self.n_b)
        o = self.n_bvh.order
        inv = np.empty_like(o)
        inv[o] = np.arange(len(o))

        def remap(idx):
            out = idx.copy()
            m = idx >= 0
            out[m] = inv[idx[m]]
            return out

        self.n_a, self.n_b = self.n_a[o], self.n_b[o]
        self.n_prev, self.n_next = remap(self.n_prev[o]), remap(self.n_next[o])
        d = self.n_b - self.n_a
        L = np.linalg.norm(d, axis=1, keepdims=True) if len(d) else np.ones((0, 1))
        self.n_normal = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.maximum(L, 1e-300)

        self.n_cone_axis, self.n_cone_angle = self._normal_cones()

        lo = np.minimum(self.all_a, self.all_b).min(axis=0)
        hi = np.maximum(self.all_a, self.all_b).max(axis=0)
        self.bbox = (lo, hi)
        self.scale = float(np.max(hi - lo))
        self._built = True
        return self

    def _normal_cones(self):
        """Per-BVH-node cone bounding the normals relevant to silhouette tests.

        A segment's start vertex is tested using its own normal and that of
        its predecessor, so both go into the cone. Nodes containing an open
        chain end (always a silhouette) get angle pi, i.e. never pruned.
        Nodes whose edges are all collinear get angle exactly 0: every
        (n . d) product is a square there, so they never hold a silhouette.
        """
        bvh = self.n_bvh
        m = len(bvh.lo)
        axis, angle = np.zeros((m, 2)), np.full(m, np.pi)
        has_prev = self.n_prev >= 0
        open_end = (~has_prev) | (self.n_next < 0)
        for node in range(m):
            s, e = bvh.start[node], bvh.start[node] + bvh.count[node]
            if open_end[s:e].any():
                continue
            n = np.concatenate([self.n_normal[s:e], self.n_normal[self.n_prev[s:e]]])
            mean = n.mean(axis=0)
            L = np.linalg.norm(mean)
            if L < 1e-6:
                continue
            axis[node] = mean / L
            if np.abs(n - n[0]).max() < 1e-12:
                angle[node] = 0.0  # collinear chain: never has a silhouette
            else:
                angle[node] = np.arccos(np.clip(n @ axis[node], -1, 1)).max()
        return axis, angle

    def inside(self, points):
        """Even-odd point-in-domain test for an (m, 2) array of points."""
        from ._kernels import inside_kernel
        self.build()
        return inside_kernel(np.ascontiguousarray(points, dtype=np.float64),
                             self.all_a, self.all_b)

    def grid(self, res, pad=0.0):
        """Pixel-centered grid over the bounding box. Returns (X, Y, points, mask).

        Sampling pixel centers (not edges) keeps grid points off the boundary
        of axis-aligned domains, where estimates are undefined.
        """
        self.build()
        lo, hi = self.bbox
        lo, hi = lo - pad, hi + pad
        aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])
        nx, ny = res, max(2, int(round(res * aspect)))
        xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
        ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X.ravel(), Y.ravel()], axis=1)
        mask = self.inside(P)
        return X, Y, P, mask


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------

def circle(center, radius, n=256):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)


def rectangle(x0, y0, x1, y1, n_per_side=1):
    """Rectangle as CCW loop, each side split into n_per_side segments.

    Side order of the segments: bottom, right, top, left.
    """
    t = np.linspace(0, 1, n_per_side, endpoint=False)
    bottom = np.stack([x0 + (x1 - x0) * t, np.full_like(t, y0)], 1)
    right = np.stack([np.full_like(t, x1), y0 + (y1 - y0) * t], 1)
    top = np.stack([x1 - (x1 - x0) * t, np.full_like(t, y1)], 1)
    left = np.stack([np.full_like(t, x0), y1 - (y1 - y0) * t], 1)
    return np.concatenate([bottom, right, top, left])


def koch_snowflake(level, radius=1.0):
    pts = circle((0, 0), radius, 3)
    for _ in range(level):
        new = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            d = (b - a) / 3
            p1, p3 = a + d, a + 2 * d
            # Bump outward: rotate d by -60 degrees for a CCW polygon.
            c, s = 0.5, -np.sqrt(3) / 2
            p2 = p1 + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
            new.extend([a, p1, p2, p3])
        pts = np.array(new)
    return pts
