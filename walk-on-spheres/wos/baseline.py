"""Baseline Walk on Spheres in plain NumPy.

This is the "textbook" algorithm from Muller (1956) / Sawhney & Crane (2020),
kept deliberately simple so it can serve as a reference and as the starting
point for the improvements in `wos.solver`:

  * Dirichlet boundaries only.
  * Closest-point queries by brute force over every segment: O(n_segments).
  * Poisson source term sampled *uniformly* in each ball.
  * Fixed number of walks per point, NumPy global RNG.

Convention: solves  Laplacian(u) = -f  in the domain,  u = g  on the boundary.
"""

import numpy as np


def _closest_dirichlet(P, a, b, ga, gb):
    """Distance from each point in P to the nearest Dirichlet segment, and g there."""
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    ap = P[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("mnk,nk->mn", ap, ab) / ab2, 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    d2 = np.sum((P[:, None, :] - q) ** 2, axis=-1)
    j = np.argmin(d2, axis=1)
    rows = np.arange(len(P))
    tj = t[rows, j]
    return np.sqrt(d2[rows, j]), ga[j] + tj * (gb[j] - ga[j])


def solve(scene, points, n_walks=256, eps=1e-4, max_steps=10_000,
          source=None, rng=None):
    """Estimate u at each point.

    eps is relative to the scene size. `source` is a vectorized f(x, y).
    Returns (mean, standard_error) arrays of shape (n_points,).
    """
    scene.build()
    if len(scene.n_a):
        raise ValueError("baseline solver supports Dirichlet boundaries only")
    rng = np.random.default_rng() if rng is None else rng
    points = np.asarray(points, dtype=np.float64)
    m = len(points)
    eps_abs = eps * scene.scale
    nseg = len(scene.d_a)

    est = np.zeros(m * n_walks)
    owner = np.repeat(np.arange(m), n_walks)
    X = points[owner].copy()

    # Process walkers in chunks so the (walkers x segments) matrix fits in memory.
    chunk = max(1, int(4e6 // nseg))
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        acc = np.zeros(len(x))
        active = np.arange(len(x))
        for _ in range(max_steps):
            if len(active) == 0:
                break
            p = x[active]
            R, g = _closest_dirichlet(p, scene.d_a, scene.d_b, scene.d_ga, scene.d_gb)
            done = R < eps_abs
            acc[active[done]] += g[done]
            active, p, R = active[~done], p[~done], R[~done]
            if len(active) == 0:
                break
            if source is not None:
                # Uniform sample y in the ball; weight |B| * G(x, y).
                r = R * np.sqrt(rng.random(len(R)))
                th = 2 * np.pi * rng.random(len(R))
                y = p + r[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
                G = np.log(R / np.maximum(r, 1e-300)) / (2 * np.pi)
                acc[active] += np.pi * R ** 2 * source(y[:, 0], y[:, 1]) * G
            th = 2 * np.pi * rng.random(len(R))
            x[active] = p + R[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
        else:
            # Walks that hit max_steps: use the closest boundary value (biased).
            if len(active):
                _, g = _closest_dirichlet(x[active], scene.d_a, scene.d_b,
                                          scene.d_ga, scene.d_gb)
                acc[active] += g
        est[s:s + chunk] = acc

    est = est.reshape(m, n_walks)
    return est.mean(axis=1), est.std(axis=1, ddof=1) / np.sqrt(n_walks)
