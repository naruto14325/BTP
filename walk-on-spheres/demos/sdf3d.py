"""3D Laplace on an SDF domain: a hot torus inside a cold box, sliced at z = 0
and y = 0. No volume mesh is ever built.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

import wos
from _plot import save


@njit
def box(x, y, z):
    qx, qy, qz = abs(x) - 1.0, abs(y) - 1.0, abs(z) - 1.0
    out = np.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
    return out + min(max(qx, max(qy, qz)), 0.0)


@njit
def torus(x, y, z):
    q = np.sqrt(x * x + y * y) - 0.55
    return np.sqrt(q * q + z * z) - 0.15


@njit
def sdf(x, y, z):
    return max(box(x, y, z), -torus(x, y, z))  # inside box, outside torus


@njit
def g(x, y, z):
    return 1.0 if abs(torus(x, y, z)) < abs(box(x, y, z)) else 0.0


n = 300
s = np.linspace(-1, 1, n)
A, B = np.meshgrid(s, s)
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, P) in zip(axs, [
        ("slice z = 0", np.stack([A.ravel(), B.ravel(), 0 * A.ravel()], 1)),
        ("slice y = 0", np.stack([A.ravel(), 0 * A.ravel(), B.ravel()], 1))]):
    inside = np.array([sdf(*p) < 0 for p in P])
    t0 = time.time()
    U = np.full(len(P), np.nan)
    U[inside], _ = wos.solve_sdf3(P[inside], sdf, g, 256, eps=1e-3)
    print(f"{name}: {inside.sum()} points in {time.time() - t0:.1f}s")
    U = U.reshape(A.shape)
    ax.pcolormesh(A, B, U, cmap="inferno", shading="auto", rasterized=True)
    ax.contour(A, B, U, levels=12, colors="white", linewidths=0.4, alpha=0.6)
    ax.set_title(name); ax.set_aspect("equal"); ax.axis("off")
save(fig, "sdf3d.png")
