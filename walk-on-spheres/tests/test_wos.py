"""Validate the solvers against problems with known closed-form solutions.

Monte Carlo estimates are random, so most checks are z-scores: the error
divided by the estimator's own standard error should look like N(0, 1).
Seeds are fixed, so the tests are deterministic.
"""

import numpy as np
import pytest
from numba import njit

import wos
from wos import NEUMANN, DIRICHLET, baseline
from wos import _kernels as K


def cubic(x, y):
    return x ** 3 - 3 * x * y ** 2  # Re(z^3), harmonic


@pytest.fixture(scope="module")
def square():
    return wos.Scene().add_loop(wos.rectangle(0, 0, 1, 1, 200), values=cubic)


@pytest.fixture(scope="module")
def pts():
    return np.random.default_rng(1).random((40, 2)) * 0.8 + 0.1


def assert_z(est, err, exact, zmax=4.5):
    z = np.abs(est - exact) / err
    assert z.max() < zmax, f"max z-score {z.max():.2f}"
    # The errors should also be consistent with their claimed size on average.
    assert 0.4 < np.sqrt(np.mean(z ** 2)) < 1.6


# --- geometry -----------------------------------------------------------------

def test_bvh_closest_matches_brute_force():
    sc = wos.Scene().add_loop(wos.koch_snowflake(4)).build()
    P = np.random.default_rng(0).uniform(-1.2, 1.2, (500, 2))
    fast = K.closest_dist_kernel(P, sc.d_a, sc.d_b, *sc.d_bvh.arrays())
    brute, _ = baseline._closest_dirichlet(P, sc.d_a, sc.d_b, sc.d_ga, sc.d_gb)
    np.testing.assert_allclose(fast, brute, rtol=1e-12, atol=1e-14)


def test_inside_with_hole():
    sc = wos.Scene().add_loop(wos.rectangle(-1, -1, 1, 1)).add_loop(
        wos.circle((0, 0), 0.5), hole=True)
    P = np.array([[0, 0], [0.75, 0], [0, 0.9], [1.5, 0]])
    assert sc.inside(P).tolist() == [False, True, True, False]


def test_loops_oriented_with_interior_on_left():
    sc = wos.Scene()
    sc.add_loop(wos.rectangle(-2, -2, 2, 2)[::-1], kinds=NEUMANN, values=0)  # given CW
    sc.add_loop(wos.circle((0, 0), 0.5, 64), kinds=NEUMANN, hole=True)
    sc.add_loop(wos.rectangle(-0.1, 1, 0.1, 1.2), kinds=DIRICHLET, hole=True)
    sc.build()
    mid = 0.5 * (sc.n_a + sc.n_b)
    assert sc.inside(mid + 1e-6 * sc.n_normal).all()
    assert not sc.inside(mid - 1e-6 * sc.n_normal).any()


# --- Dirichlet / Laplace ------------------------------------------------------

def test_laplace_dirichlet(square, pts):
    r = wos.solve(square, pts, 4096)
    assert_z(r.value, r.stderr, cubic(*pts.T))


def test_baseline_agrees(square, pts):
    m, e = baseline.solve(square, pts[:10], 1024, rng=np.random.default_rng(0))
    assert_z(m, e, cubic(*pts[:10].T))


def test_deterministic_given_seed(square, pts):
    a = wos.solve(square, pts, 64, seed=7).value
    b = wos.solve(square, pts, 64, seed=7).value
    c = wos.solve(square, pts, 64, seed=8).value
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_adaptive_sampling_reaches_tolerance(square, pts):
    r = wos.solve(square, pts, 64, tol=5e-3, max_walks=1 << 20)
    assert (r.stderr < 5e-3).all()
    assert r.walks.min() < r.walks.max()  # different points needed different budgets


# --- Poisson ------------------------------------------------------------------

@njit
def four(x, y):
    return 4.0


@njit
def bump(x, y):
    # f = -Laplacian of u = sin(pi x) sin(pi y)
    return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def test_poisson_disk():
    sc = wos.Scene().add_loop(wos.circle((0, 0), 1, 1024))
    P = np.random.default_rng(2).uniform(-0.6, 0.6, (30, 2))
    r = wos.solve(sc, P, 4096, source=four)
    assert_z(r.value, r.stderr, 1 - (P ** 2).sum(1))


def test_poisson_square_variable_source(pts):
    sc = wos.Scene().add_loop(wos.rectangle(0, 0, 1, 1, 50), values=0.0)
    r = wos.solve(sc, pts, 4096, source=bump)
    assert_z(r.value, r.stderr, np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1]))


# --- Neumann (Walk on Stars) -------------------------------------------------

def test_neumann_square(pts):
    g = lambda x, y: np.cosh(np.pi * x) * np.cos(np.pi * y)
    kinds = np.repeat([NEUMANN, DIRICHLET, NEUMANN, DIRICHLET], 200)
    sc = wos.Scene().add_loop(wos.rectangle(0, 0, 1, 1, 200), kinds=kinds, values=g)
    r = wos.solve(sc, pts, 4096)
    assert_z(r.value, r.stderr, g(*pts.T))


def test_neumann_cylinder_flow():
    """Potential flow past an insulating cylinder: u = x (1 + a^2 / r^2)."""
    a = 0.5
    g = lambda x, y: x * (1 + a ** 2 / (x ** 2 + y ** 2))
    sc = wos.Scene().add_loop(wos.rectangle(-2, -2, 2, 2, 100), values=g)
    sc.add_loop(wos.circle((0, 0), a, 512), kinds=NEUMANN, hole=True)
    rng = np.random.default_rng(3)
    th, rad = rng.uniform(0, 2 * np.pi, 30), rng.uniform(0.55, 1.5, 30)
    P = np.stack([rad * np.cos(th), rad * np.sin(th)], 1)
    r = wos.solve(sc, P, 4096)
    # Polygonizing the circle costs a little accuracy; allow a small bias.
    assert np.abs(r.value - g(*P.T)).max() < 4.5 * r.stderr.max() + 5e-3


def test_neumann_nonconvex_comb():
    """Comb-shaped domain: horizontal walls insulated, vertical walls u = x.

    u = x is the exact solution; many concave corners exercise silhouettes.
    """
    xs = [0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 0]
    ys = [0, 0, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0, 0, 1, 1]
    pts = np.array([xs, ys]).T
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    kinds = np.where(np.abs(d[:, 1]) > 0, DIRICHLET, NEUMANN)
    # Densify so the BVH has real work to do.
    sc = wos.Scene().add_loop(pts, kinds=kinds, values=lambda x, y: x)
    P = np.array([[0.1, 0.5], [0.5, 0.5], [0.3, 0.9], [0.9, 0.5], [0.5, 0.95], [0.7, 0.1]])
    r = wos.solve(sc, P, 8192)
    assert np.abs(r.value - P[:, 0]).max() < 4.5 * r.stderr.max() + 2e-3


# --- Gradients ---------------------------------------------------------------

def test_gradient_laplace(square, pts):
    u, G = wos.gradient(square, pts, 8192)
    x, y = pts.T
    exact = np.stack([3 * x ** 2 - 3 * y ** 2, -6 * x * y], 1)
    assert np.abs(u - cubic(x, y)).max() < 0.05
    assert np.median(np.linalg.norm(G - exact, axis=1)) < 0.1


def test_gradient_poisson(pts):
    sc = wos.Scene().add_loop(wos.rectangle(0, 0, 1, 1, 50), values=0.0)
    _, G = wos.gradient(sc, pts, 8192, source=bump)
    x, y = pts.T
    exact = np.pi * np.stack([np.cos(np.pi * x) * np.sin(np.pi * y),
                              np.sin(np.pi * x) * np.cos(np.pi * y)], 1)
    assert np.median(np.linalg.norm(G - exact, axis=1)) < 0.1


def test_gradient_on_boundary_is_nan_not_inf(square):
    u, G = wos.gradient(square, np.array([[0.0, 0.5], [0.5, 0.5]]), 16)
    assert np.isnan(G[0]).all() and np.isfinite(G[1]).all()


def test_grid_points_avoid_walls():
    sc = wos.Scene().add_loop(wos.rectangle(0, 0, 4, 1.5, 10))
    _, _, P, mask = sc.grid(120)
    d = K.closest_dist_kernel(P[mask], sc.d_a, sc.d_b, *sc.d_bvh.arrays())
    assert d.min() > 1e-3


def test_control_variate_reduces_variance():
    # Adding a constant to u does not change grad u. Without the control
    # variate the estimator's variance grows with that constant; with it, not.
    sc = wos.Scene().add_loop(wos.rectangle(0, 0, 1, 1, 200),
                              values=lambda x, y: cubic(x, y) + 10)
    P = np.full((200, 2), 0.5)
    _, a = wos.gradient(sc, P, 64, control_variate=False, seed=1)
    _, b = wos.gradient(sc, P, 64, control_variate=True, seed=1)
    assert b.var(axis=0).sum() < 0.05 * a.var(axis=0).sum()


# --- 3D ----------------------------------------------------------------------

@njit
def ball_sdf(x, y, z):
    return np.sqrt(x * x + y * y + z * z) - 1.0


@njit
def point_charge(x, y, z):
    # 1/|p - c| with c outside the ball is harmonic inside it.
    return 1.0 / np.sqrt((x - 2.0) ** 2 + y * y + z * z)


def test_sdf3_ball():
    P = np.random.default_rng(4).uniform(-0.5, 0.5, (20, 3))
    m, e = wos.solve_sdf3(P, ball_sdf, point_charge, 4096)
    exact = 1 / np.sqrt((P[:, 0] - 2) ** 2 + P[:, 1] ** 2 + P[:, 2] ** 2)
    assert np.abs(m - exact).max() < 4.5 * e.max() + 1e-3


def test_normal_cone_pruning_is_exact():
    """Cone pruning must skip only nodes that cannot hold a silhouette."""
    sc = wos.Scene().add_loop(wos.rectangle(-2, -2, 2, 2, 40), values=1.0)
    sc.add_loop(wos.circle((0, 0), 0.5, 2048), kinds=NEUMANN, hole=True)
    sc.add_loop(wos.koch_snowflake(3, 0.4) + [1.2, 1.2], kinds=NEUMANN, hole=True)
    sc.build()
    P = np.random.default_rng(5).uniform(-2, 2, (2000, 2))
    args = (sc.n_a, sc.n_b, sc.n_prev, sc.n_next)
    stack = np.empty(64, np.int64)
    no_cones = np.full_like(sc.n_cone_angle, np.pi)
    for x, y in P:
        a = K.closest_silhouette(x, y, *args, sc.n_cone_axis, sc.n_cone_angle,
                                 *sc.n_bvh.arrays(), stack, K.INF)
        b = K.closest_silhouette(x, y, *args, sc.n_cone_axis, no_cones,
                                 *sc.n_bvh.arrays(), stack, K.INF)
        assert a == b


def test_gradient_stderr_is_calibrated(square, pts):
    u, G, ue, ge = wos.gradient(square, pts, 4096, stderr=True)
    x, y = pts.T
    exact = np.stack([3 * x ** 2 - 3 * y ** 2, -6 * x * y], 1)
    assert_z(u, ue, cubic(x, y))
    assert_z(G.ravel(), ge.ravel(), exact.ravel())


# --- Boundary value caching --------------------------------------------------

def test_cache_laplace_value_and_gradient(square):
    P = np.random.default_rng(6).random((300, 2)) * 0.9 + 0.05
    P = np.vstack([P, [[0.01, 0.5], [0.5, 0.99], [0.015, 0.015]]])  # near the wall
    C = wos.BoundaryCache(square, 2000, 1024, delta=0.02)
    r = C.evaluate(P, gradient=True)
    x, y = P.T
    assert r.cached.mean() > 0.9 and not r.cached[-3:].any()
    c = r.cached
    assert_z(r.value[c], r.stderr[c], cubic(x, y)[c], zmax=5)
    exact = np.stack([3 * x ** 2 - 3 * y ** 2, -6 * x * y], 1)
    err = np.linalg.norm(r.grad - exact, axis=1)
    assert np.median(err[c]) < 0.03
    # Points near the wall fall back to direct walks and still agree.
    assert np.abs(r.value[~c] - cubic(x, y)[~c]).max() < 5 * r.stderr[~c].max() + 1e-3


def test_cache_neumann_cylinder():
    a = 0.5
    g = lambda x, y: x * (1 + a ** 2 / (x ** 2 + y ** 2))
    sc = wos.Scene().add_loop(wos.rectangle(-2, -2, 2, 2, 100), values=g)
    sc.add_loop(wos.circle((0, 0), a, 512), kinds=NEUMANN, hole=True)
    rng = np.random.default_rng(7)
    th, rad = rng.uniform(0, 2 * np.pi, 200), rng.uniform(0.6, 1.4, 200)
    P = np.stack([rad * np.cos(th), rad * np.sin(th)], 1)
    r = wos.BoundaryCache(sc, 4000, 1024, delta=0.01).evaluate(P, gradient=True)
    x, y = P.T
    r2 = x ** 2 + y ** 2
    exact = np.stack([1 + a ** 2 * (y ** 2 - x ** 2) / r2 ** 2, -2 * a ** 2 * x * y / r2 ** 2], 1)
    assert np.abs(r.value - g(x, y)).max() < 0.06
    assert np.median(np.linalg.norm(r.grad - exact, axis=1)) < 0.03


def test_cache_poisson(pts):
    sc = wos.Scene().add_loop(wos.rectangle(0, 0, 1, 1, 50), values=0.0)
    r = wos.BoundaryCache(sc, 2000, 1024, delta=0.02, source=bump).evaluate(pts)
    exact = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
    assert np.abs(r.value - exact).max() < 0.03


def test_cache_rejects_delta_larger_than_features():
    sc = wos.Scene().add_loop(wos.koch_snowflake(4))
    with pytest.raises(ValueError):
        wos.BoundaryCache(sc, 500, 16, delta=0.05)
