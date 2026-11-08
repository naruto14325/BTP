"""Fast grid-free PDE solver: Walk on Spheres / Walk on Stars (Numba).

Solves   Laplacian(u) = -f   in a 2D domain, with
         u = g               on Dirichlet segments,
         du/dn = 0           on Neumann segments.

Compared with `wos.baseline` this adds:
  * BVH-accelerated geometric queries (O(log n) per step instead of O(n)),
  * Numba + multithreading, with per-point counter-based RNG streams,
  * Walk on Stars for (zero-flux) Neumann boundaries,
  * importance-sampled source term (bounded weights, lower variance),
  * adaptive sampling to a target standard error,
  * gradient estimation with a control variate,
  * a 3D solver for domains given by signed distance functions.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from . import _kernels as K


@njit
def _zero_source(x, y):
    return 0.0


@dataclass
class Result:
    value: np.ndarray        # estimated u at each point
    stderr: np.ndarray       # standard error of each estimate
    walks: np.ndarray        # number of walks used per point
    steps: np.ndarray        # mean walk length per point


def _scene_args(scene):
    scene.build()
    d = (scene.d_a, scene.d_b, scene.d_ga, scene.d_gb) + scene.d_bvh.arrays()
    n = (scene.n_a, scene.n_b, scene.n_prev, scene.n_next, scene.n_normal,
         scene.n_cone_axis, scene.n_cone_angle) + scene.n_bvh.arrays()
    return d + n


def _common(scene, source, source_samples, eps, rmin, max_steps):
    has_source = source is not None
    src = source if has_source else _zero_source
    return (src, has_source, int(source_samples), eps * scene.scale, rmin * scene.scale,
            1e-9 * scene.scale, int(max_steps))


def solve(scene, points, n_walks=256, *, tol=0.0, max_walks=None, source=None,
          source_samples=1, eps=1e-4, rmin=1e-4, max_steps=100_000, seed=0):
    """Estimate u at the given (m, 2) points.

    n_walks   : walks per point (the minimum if `tol` is set).
    tol       : if > 0, keep adding walks until stderr < tol or max_walks.
    source    : numba-jitted f(x, y) for the Poisson term, or None.
    source_samples : source-term samples per walk step (helps when f is
                localized: they share the cost of one walk).
    eps, rmin : Dirichlet shell width and minimum star radius, relative to
                the scene size.
    """
    P = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    n_max = n_walks if tol <= 0 else (max_walks or 64 * n_walks)
    mean, err, used, steps = K.solve_kernel(
        P, int(n_walks), int(n_max), float(tol), int(seed),
        *_scene_args(scene), *_common(scene, source, source_samples, eps, rmin, max_steps))
    return Result(mean, err, used, steps)


def gradient(scene, points, n_walks=256, *, source=None, source_samples=1,
             control_variate=True, eps=1e-4, rmin=1e-4, max_steps=100_000, seed=0,
             stderr=False):
    """Estimate (u, grad u) at the given points. Returns (u, grad) arrays,
    or (u, grad, u_stderr, grad_stderr) if stderr=True.

    Points within eps of the boundary get NaN: the first step needs a ball
    that touches no boundary, and its radius sets the variance (~1 / R^2).
    """
    P = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    u, g, ue, ge = K.gradient_kernel(
        P, int(n_walks), int(seed), bool(control_variate),
        *_scene_args(scene), *_common(scene, source, source_samples, eps, rmin, max_steps))
    return (u, g, ue, ge) if stderr else (u, g)


def solve_sdf3(points, sdf, g, n_walks=256, *, eps=1e-4, max_steps=100_000, seed=0):
    """3D Laplace with Dirichlet data g on the zero set of `sdf` (both numba-jitted)."""
    P = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    mean, err = K.solve_sdf3_kernel(P, int(n_walks), int(seed), sdf, g, float(eps), int(max_steps))
    return mean, err


def solve_grid(scene, res=256, n_walks=128, **kw):
    """Solve on a res-wide grid over the scene's bounding box.

    Returns (X, Y, U) with U = NaN outside the domain, plus the Result.
    """
    X, Y, P, mask = scene.grid(res)
    U = np.full(len(P), np.nan)
    r = solve(scene, P[mask], n_walks, **kw)
    U[mask] = r.value
    return X, Y, U.reshape(X.shape), r
