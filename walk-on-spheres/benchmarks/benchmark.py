"""Measure each improvement over the baseline. Prints a Markdown report.

    PYTHONPATH=. python benchmarks/benchmark.py > benchmarks/results.md
"""

import time

import numpy as np
from numba import njit

import wos
from wos import NEUMANN, baseline


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def cubic(x, y):
    return x ** 3 - 3 * x * y ** 2


@njit
def four(x, y):
    return 4.0


print("# Benchmarks\n")
print(f"Machine: {__import__('platform').processor() or 'unknown'}, "
      f"{__import__('os').cpu_count()} threads. Times are wall-clock seconds.\n")

# --- 1. Speed: NumPy brute force vs Numba + BVH -----------------------------
print("## 1. Speed: baseline (NumPy, brute force) vs fast (Numba + BVH, multithreaded)\n")
print("| domain | segments | baseline walks/s | fast walks/s | speedup |")
print("|---|---|---|---|---|")
for name, pts in [("square", wos.rectangle(0, 0, 1, 1, 50)),
                  ("square", wos.rectangle(0, 0, 1, 1, 200)),
                  ("Koch snowflake L4", wos.koch_snowflake(4)),
                  ("Koch snowflake L5", wos.koch_snowflake(5))]:
    sc = wos.Scene().add_loop(pts, values=lambda x, y: x).build()
    P = np.random.default_rng(0).uniform(-0.3, 0.3, (8, 2)) + (0.5 if name == "square" else 0)
    nb = 256
    _, tb = timed(baseline.solve, sc, P, nb, rng=np.random.default_rng(0))
    wos.solve(sc, P[:1], 4)
    nf = 1 << 15
    _, tf = timed(wos.solve, sc, P, nf)
    bw, fw = len(P) * nb / tb, len(P) * nf / tf
    print(f"| {name} | {len(pts)} | {bw:,.0f} | {fw:,.0f} | **{fw / bw:,.0f}x** |")

# --- 2. Importance-sampled source term --------------------------------------
print("\n## 2. Poisson source term: uniform sampling (baseline) vs importance sampling\n")
print("Per-walk standard deviation of the estimate (lower is better); "
      "Δu = -4 on the unit disk.\n")
sc = wos.Scene().add_loop(wos.circle((0, 0), 1, 512)).build()
P = np.array([[0.3, 0.1], [0.6, -0.2], [-0.2, 0.75], [0.0, -0.9]])
_, eb = baseline.solve(sc, P, 4096, source=lambda x, y: 4.0 + 0 * x, rng=np.random.default_rng(1))
rf = wos.solve(sc, P, 4096, source=four)
sb, sf = eb * 64, rf.stderr * 64
print("| point | uniform σ | importance σ | variance reduction |")
print("|---|---|---|---|")
for p, a, b in zip(P, sb, sf):
    print(f"| ({p[0]:.1f}, {p[1]:.1f}) | {a:.3f} | {b:.3f} | **{(a / b) ** 2:.1f}x** |")

# --- 3. Gradient control variate --------------------------------------------
print("\n## 3. Gradient estimation: control variate\n")
print("Variance of ∇u at (0.5, 0.5) with 64 walks, repeated 400 times. "
      "u = Re(z³) + c; adding c does not change ∇u.\n")
print("| offset c | no control variate | with control variate | reduction |")
print("|---|---|---|---|")
for c in [0.0, 1.0, 10.0]:
    sc = wos.Scene().add_loop(wos.rectangle(0, 0, 1, 1, 200), values=lambda x, y: cubic(x, y) + c)
    P = np.full((400, 2), 0.5)
    _, a = wos.gradient(sc, P, 64, control_variate=False, seed=1)
    _, b = wos.gradient(sc, P, 64, control_variate=True, seed=1)
    va, vb = a.var(0).sum(), b.var(0).sum()
    print(f"| {c:g} | {va:.3f} | {vb:.3f} | **{va / vb:.1f}x** |")

# --- 4. Adaptive sampling ---------------------------------------------------
print("\n## 4. Adaptive sampling (Koch snowflake L5, 2000 random points)\n")
sc = wos.Scene().add_loop(wos.koch_snowflake(5), values=lambda x, y: np.cos(5 * np.arctan2(y, x)))
sc.build()
P = np.random.default_rng(2).uniform(-1, 1, (4000, 2))
P = P[sc.inside(P)][:2000]
tol = 0.02
pilot = wos.solve(sc, P, 1024, seed=9)
sigma = pilot.stderr * np.sqrt(1024)
n_uniform = int(np.ceil((sigma.max() / tol) ** 2))
r, ta = timed(wos.solve, sc, P, 64, tol=tol, max_walks=1 << 20)
print(f"Target: standard error < {tol} at every point.\n")
print("| strategy | total walks | time |")
print("|---|---|---|")
_, tu = timed(wos.solve, sc, P[:200], n_uniform)
print(f"| same count everywhere ({n_uniform:,} per point) | {n_uniform * len(P):,} | "
      f"{tu * len(P) / 200:.1f} s (est.) |")
print(f"| adaptive (64 minimum, batches of 32) | {r.walks.sum():,} | {ta:.1f} s |")
print(f"\n**{n_uniform * len(P) / r.walks.sum():.1f}x fewer walks**; "
      f"max stderr achieved {r.stderr.max():.4f}.")

# --- 5. Normal-cone pruning for silhouette queries --------------------------
print("\n## 5. Silhouette queries: normal-cone pruning (Walk on Stars)\n")
print("Two insulated circles in a box, 400 points x 512 walks. Results are bit-identical.\n")
print("| segments per circle | without cones | with cones | speedup |")
print("|---|---|---|---|")
a = 0.5
g = lambda x, y: x * (1 + a ** 2 / (x ** 2 + y ** 2))
for nseg in [256, 4096, 16384]:
    sc = wos.Scene().add_loop(wos.rectangle(-2, -2, 2, 2, 100), values=g)
    sc.add_loop(wos.circle((0, 0), a, nseg), kinds=NEUMANN, hole=True)
    sc.add_loop(wos.circle((1.2, 1.2), 0.3, nseg), kinds=NEUMANN, hole=True)
    sc.build()
    P = np.random.default_rng(0).uniform(-1.9, 1.9, (400, 2))
    P = P[sc.inside(P)]
    wos.solve(sc, P[:1], 4)
    r1, t1 = timed(wos.solve, sc, P, 512)
    saved = sc.n_cone_angle.copy()
    sc.n_cone_angle[:] = np.pi
    r2, t2 = timed(wos.solve, sc, P, 512)
    sc.n_cone_angle[:] = saved
    assert np.array_equal(r1.value, r2.value)
    print(f"| {nseg} | {t2:.2f} s | {t1:.2f} s | **{t2 / t1:.1f}x** |")

# --- 6. Several source samples per walk step ---------------------------------
print("\n## 6. Localized sources: several source samples per walk step\n")
print("Heat demo scene (two narrow Gaussian heaters, mixed boundaries), 5 points x 8192 walks. "
      "Efficiency = 1 / (variance x time), relative to K = 1.\n")
import os, sys  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demos"))
src = open(os.path.join(os.path.dirname(__file__), "..", "demos", "heat.py")).read()
ns = {}
exec(src.split("t0 = time.time()")[0], ns)
scene, heaters = ns["scene"], ns["heaters"]
P = np.array([[0.8, 2.0], [1.5, 1.0], [3.0, 0.9], [0.5, 0.4], [1.0, 2.8]])
wos.solve(scene, P[:1], 4, source=heaters)
print("| samples per step K | time | mean per-walk σ | relative efficiency |")
print("|---|---|---|---|")
base = None
for k in [1, 2, 4, 8, 16]:
    r, t = timed(wos.solve, scene, P, 8192, source=heaters, source_samples=k)
    var = (r.stderr ** 2 * 8192).mean()
    eff = 1 / (var * t)
    base = base or eff
    print(f"| {k} | {t:.2f} s | {np.sqrt(var):.2f} | **{eff / base:.2f}x** |")

# --- 7. Boundary value caching ----------------------------------------------
print("\n## 7. Boundary value caching vs direct walks (full image)\n")
print("Potential flow past an insulated cylinder (analytic solution known), "
      "evaluated on a 200 x 200 grid. Errors are against the exact u and ∇u.\n")
a = 0.5
g = lambda x, y: x * (1 + a ** 2 / (x ** 2 + y ** 2))
sc = wos.Scene().add_loop(wos.rectangle(-2, -2, 2, 2, 100), values=g)
sc.add_loop(wos.circle((0, 0), a, 512), kinds=NEUMANN, hole=True)
X, Y, P, mask = sc.grid(200)
P = P[mask]
x, y = P.T
r2 = x ** 2 + y ** 2
ue = g(x, y)
ge = np.stack([1 + a ** 2 * (y ** 2 - x ** 2) / r2 ** 2, -2 * a ** 2 * x * y / r2 ** 2], 1)


def errs(u, G):
    eu = np.abs(u - ue)
    eg = np.linalg.norm(G - ge, axis=1)
    return np.sqrt(np.mean(eu ** 2)), np.nanmedian(eg), np.nanpercentile(eg, 95)


print("| method | walks | time | u RMS error | ∇u median error | ∇u 95th pct error |")
print("|---|---|---|---|---|---|")
wos.solve(sc, P[:1], 4); wos.gradient(sc, P[:1], 4)
for n in [256, 1024]:
    (u, G), t = timed(wos.gradient, sc, P, n)
    bad = np.isnan(u)
    u[bad] = wos.solve(sc, P[bad], n).value
    eu, gm, g95 = errs(u, G)
    print(f"| direct, {n} walks/point | {n * len(P):,} | {t:.1f} s | {eu:.4f} | {gm:.4f} | {g95:.3f} |")
for M, n in [(2000, 512), (4000, 1024)]:
    t0 = time.perf_counter()
    C = wos.BoundaryCache(sc, M, n, delta=0.01)
    tb = time.perf_counter() - t0
    r, te = timed(C.evaluate, P, gradient=True, n_walks_near=256)
    walks = C.walks + 256 * (~r.cached).sum()
    eu, gm, g95 = errs(r.value, r.grad)
    print(f"| **cache**, M={M}, {n} walks/cache pt ({r.cached.mean():.0%} cached) | {walks:,} | "
          f"{tb + te:.1f} s ({tb:.1f} build) | {eu:.4f} | {gm:.4f} | {g95:.3f} |")
