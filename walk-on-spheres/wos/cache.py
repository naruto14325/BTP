"""Boundary Value Caching (Miller, Sawhney, Crane & Gkioulekas 2023).

Instead of running walks from every query point, run them only from M
cache points y_j on a curve offset by delta inside the boundary, estimating
u(y_j) and the outward normal derivative q_j = du/dn(y_j) there. For any
x inside that offset curve, Green's representation formula

    u(x) = sum_j w_j [ G(x, y_j) q_j - u_j dG/dn_y(x, y_j) ] + int G(x, y) f(y) dy,

    G(x, y) = -log|x - y| / (2 pi),

reconstructs the solution from the cache in O(M) cheap kernel evaluations,
with no walks at all. Differentiating the kernels in x gives grad u
analytically. Points within a few cache spacings of the boundary, where the
quadrature is inaccurate, fall back to direct Walk on Stars.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from . import _kernels as K
from .geometry import NEUMANN
from .solver import gradient as wos_gradient
from .solver import solve as wos_solve

TWO_PI = 2.0 * math.pi


@dataclass
class CacheResult:
    value: np.ndarray      # u at each point
    stderr: np.ndarray     # standard error (propagated from the cache, or direct WoS)
    grad: np.ndarray       # grad u, (m, 2), or None
    cached: np.ndarray     # True where the cache was used, False for direct walks


@njit(parallel=True, cache=True)
def _eval_source(source, P):
    out = np.empty(P.shape[0])
    for i in prange(P.shape[0]):
        out[i] = source(P[i, 0], P[i, 1])
    return out


@njit(parallel=True, cache=True, error_model="numpy")
def _bvc_kernel(X, Y, N, W, U, Q, UE, QE, V, VWF, a, want_grad):
    m, M = X.shape[0], Y.shape[0]
    val = np.empty(m)
    err = np.empty(m)
    grad = np.zeros((m, 2))
    for i in prange(m):
        x, y = X[i, 0], X[i, 1]
        # Singularity subtraction: the double layer of a constant is exactly
        # that constant inside the curve, so write u = c + D[u - c] + S[q]
        # with c the nearest cached value. This keeps the integrand small
        # near x and cuts quadrature error close to the boundary.
        best, jc = 1e300, 0
        for j in range(M):
            d2 = (Y[j, 0] - x) ** 2 + (Y[j, 1] - y) ** 2
            if d2 < best:
                best, jc = d2, j
        c = U[jc]
        acc, var, gx, gy = c, 0.0, 0.0, 0.0
        for j in range(M):
            dx, dy = Y[j, 0] - x, Y[j, 1] - y
            r2 = dx * dx + dy * dy
            nx, ny = N[j, 0], N[j, 1]
            dn = dx * nx + dy * ny
            G = -0.5 * math.log(r2) / TWO_PI
            Kd = -dn / (TWO_PI * r2)               # dG/dn_y
            du = U[j] - c
            acc += W[j] * (G * Q[j] - du * Kd)
            var += W[j] ** 2 * ((G * QE[j]) ** 2 + (Kd * UE[j]) ** 2)
            if want_grad:
                # grad_x G = (y - x) / (2 pi r^2)
                # grad_x dG/dn = -(1/2pi) (-n / r^2 + 2 (dn) (y - x) / r^4)
                gGx, gGy = dx / (TWO_PI * r2), dy / (TWO_PI * r2)
                gKx = -(-nx / r2 + 2.0 * dn * dx / (r2 * r2)) / TWO_PI
                gKy = -(-ny / r2 + 2.0 * dn * dy / (r2 * r2)) / TWO_PI
                gx += W[j] * (gGx * Q[j] - du * gKx)
                gy += W[j] * (gGy * Q[j] - du * gKy)
        # Volume (source) term on cells of area pi a^2, using the kernel
        # averaged over a disk of radius a to tame the log singularity.
        a2 = a * a
        for k in range(V.shape[0]):
            dx, dy = V[k, 0] - x, V[k, 1] - y
            r2 = dx * dx + dy * dy
            if r2 > a2:
                acc += VWF[k] * (-0.5 * math.log(r2) / TWO_PI)
                if want_grad:
                    gx += VWF[k] * dx / (TWO_PI * r2)
                    gy += VWF[k] * dy / (TWO_PI * r2)
            else:
                acc += VWF[k] * (-(math.log(a) - 0.5 * (1.0 - r2 / a2)) / TWO_PI)
                if want_grad:
                    gx += VWF[k] * dx / (TWO_PI * a2)
                    gy += VWF[k] * dy / (TWO_PI * a2)
        val[i] = acc
        err[i] = math.sqrt(var)
        grad[i, 0] = gx
        grad[i, 1] = gy
    return val, err, grad


def _merge_collinear(pts, kinds):
    """Drop vertices between collinear edges of the same kind.

    Offsetting shortens edges at convex corners, so many short collinear
    pieces (e.g. a finely split square side) would vanish; one long edge
    does not.
    """
    keep = np.ones(len(pts), bool)
    d = np.roll(pts, -1, axis=0) - pts
    dp = np.roll(d, 1, axis=0)
    cross = dp[:, 0] * d[:, 1] - dp[:, 1] * d[:, 0]
    dot = np.sum(dp * d, axis=1)
    flat = (np.abs(cross) <= 1e-12 * np.linalg.norm(dp, axis=1) * np.linalg.norm(d, axis=1)) & (dot > 0)
    keep &= ~(flat & (kinds == np.roll(kinds, 1)))
    return pts[keep], kinds[keep]


def _offset_loop(pts, delta):
    """Offset a closed polygon (interior on the left) inward by delta.

    Vertices move along the miter direction, so every offset edge stays
    parallel to its original at distance exactly delta.
    """
    d = np.roll(pts, -1, axis=0) - pts
    L = np.linalg.norm(d, axis=1)
    n_in = np.stack([-d[:, 1], d[:, 0]], 1) / L[:, None]
    n_prev = np.roll(n_in, 1, axis=0)
    denom = 1.0 + np.sum(n_prev * n_in, axis=1)
    if (denom < 1e-6).any():
        raise ValueError("boundary has a hairpin turn; cannot offset it")
    off = pts + delta * (n_prev + n_in) / denom[:, None]
    od = np.roll(off, -1, axis=0) - off
    if (np.sum(od * d, axis=1) <= 0).any():
        raise ValueError(f"delta={delta:g} is larger than the smallest boundary feature")
    return off, n_in


class BoundaryCache:
    """Solve once on a cache curve, then evaluate u (and grad u) anywhere.

    n_points     : number of cache points M along the (offset) boundary.
    n_walks      : walks per cache point.
    delta        : offset of the cache curve, relative to scene size. Larger
                   delta means smaller variance in du/dn (it scales like
                   1/delta) but a wider band near the boundary that needs
                   direct walks.
    source       : numba-jitted f(x, y) or None (same as wos.solve).
    volume_res   : resolution of the cell grid used for the source integral.
    neumann_flux : "zero" uses the known du/dn = 0 on cache points next to
                   Neumann walls (exact on the wall, O(delta) off it, and
                   noise-free); "estimate" estimates it with walks.
    near         : points closer than delta + near * spacing to the boundary
                   use direct walks. Values stay accurate right up to the
                   cache curve (thanks to singularity subtraction); gradients
                   lose accuracy within ~5 spacings of it, but are still far
                   less noisy there than the direct gradient estimator.
    """

    def __init__(self, scene, n_points=4000, n_walks=1024, *, delta=0.01, source=None,
                 source_samples=1, volume_res=400, neumann_flux="zero", near=2.0,
                 seed=0, **walk_kw):
        scene.build()
        self.scene, self.source, self.source_samples = scene, source, source_samples
        self.walk_kw = walk_kw
        self.delta = delta * scene.scale
        self.near = near

        loops = [_merge_collinear(p, k) for p, k, _ in scene._loops]
        offs = [(_offset_loop(p, self.delta), k) for p, k in loops]
        perims = [np.linalg.norm(np.roll(o, -1, 0) - o, axis=1) for (o, _), _ in offs]
        total = sum(L.sum() for L in perims)
        Y, Nrm, W, kinds = [], [], [], []
        for ((off, n_in), kind), L in zip(offs, perims):
            P = L.sum()
            n = max(8, int(round(n_points * P / total)))
            s = (np.arange(n) + 0.5) * P / n
            cum = np.concatenate([[0], np.cumsum(L)])
            e = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1)
            t = (s - cum[e]) / L[e]
            nxt = (e + 1) % len(off)
            Y.append(off[e] + t[:, None] * (off[nxt] - off[e]))
            Nrm.append(-n_in[e])                 # outward normal of the domain
            W.append(np.full(n, P / n))
            kinds.append(kind[e])
        self.Y = np.concatenate(Y)
        self.N = np.concatenate(Nrm)
        self.W = np.concatenate(W)
        self.kinds = np.concatenate(kinds)
        self.spacing = self.W.max()

        dist = K.closest_dist_kernel(self.Y, *self._all_segments())
        if dist.min() < 0.5 * self.delta:
            raise ValueError("cache curve passes too close to the boundary; reduce delta")

        u, g, ue, ge = wos_gradient(scene, self.Y, n_walks, source=source,
                                    source_samples=source_samples, seed=seed,
                                    stderr=True, **walk_kw)
        self.U, self.UE = u, ue
        self.Q = np.sum(g * self.N, axis=1)
        self.QE = np.sqrt(np.sum((ge * self.N) ** 2, axis=1))
        if neumann_flux == "zero":
            nm = self.kinds == NEUMANN
            self.Q[nm] = 0.0
            self.QE[nm] = 0.0
        self.walks = len(self.Y) * n_walks

        self._build_volume(volume_res)

    def _all_segments(self):
        sc = self.scene
        # A combined BVH over Dirichlet and Neumann segments, for distances.
        if not hasattr(self, "_seg"):
            from .geometry import BVH
            a, b = sc.all_a, sc.all_b
            bvh = BVH(a, b)
            self._seg = (a[bvh.order], b[bvh.order]) + bvh.arrays()
        return self._seg

    def _build_volume(self, res):
        self.V = np.zeros((0, 2))
        self.VWF = np.zeros(0)
        self.a = 1.0
        if self.source is None:
            return
        X, Y, P, mask = self.scene.grid(res)
        h = X[0, 1] - X[0, 0]
        P = P[mask]
        P = P[K.closest_dist_kernel(P, *self._all_segments()) > self.delta]
        f = _eval_source(self.source, P)
        keep = np.abs(f) > 1e-12 * np.abs(f).max()
        self.V = np.ascontiguousarray(P[keep])
        self.VWF = f[keep] * h * (Y[1, 0] - Y[0, 0])
        self.a = math.sqrt(h * (Y[1, 0] - Y[0, 0]) / math.pi)

    def evaluate(self, points, gradient=False, n_walks_near=256, seed=1):
        P = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
        dist = K.closest_dist_kernel(P, *self._all_segments())
        cached = dist > self.delta + self.near * self.spacing
        val = np.empty(len(P))
        err = np.empty(len(P))
        grad = np.empty((len(P), 2)) if gradient else None

        Pc = P[cached]
        v, e, g = _bvc_kernel(Pc, self.Y, self.N, self.W, self.U, self.Q, self.UE, self.QE,
                              self.V, self.VWF, self.a, gradient)
        val[cached], err[cached] = v, e
        if gradient:
            grad[cached] = g

        near = ~cached
        if near.any():
            kw = dict(source=self.source, source_samples=self.source_samples,
                      seed=seed, **self.walk_kw)
            if gradient:
                u, g, ue, _ = wos_gradient(self.scene, P[near], n_walks_near, stderr=True, **kw)
                bad = np.isnan(u)       # within eps of the wall: value only
                if bad.any():
                    r = wos_solve(self.scene, P[near][bad], n_walks_near, **kw)
                    u[bad], ue[bad] = r.value, r.stderr
                val[near], err[near], grad[near] = u, ue, g
            else:
                r = wos_solve(self.scene, P[near], n_walks_near, **kw)
                val[near], err[near] = r.value, r.stderr
        return CacheResult(val, err, grad, cached)
