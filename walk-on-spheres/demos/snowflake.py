"""Laplace on a Koch snowflake (3072 boundary segments), with adaptive sampling.

A grid/FEM solver would have to mesh every tiny boundary wiggle; WoS only
needs closest-point queries, so the boundary detail costs almost nothing.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import wos
from _plot import field, save

pts = wos.koch_snowflake(5)
scene = wos.Scene().add_loop(pts, values=lambda x, y: np.cos(5 * np.arctan2(y, x)))

t = time.time()
X, Y, P, mask = scene.grid(400)
r = wos.solve(scene, P[mask], 64, tol=0.02, max_walks=8192)
print(f"{len(r.value)} points, {r.walks.sum():,} walks in {time.time() - t:.1f}s")

U = np.full(X.size, np.nan); U[mask] = r.value; U = U.reshape(X.shape)
W = np.full(X.size, np.nan); W[mask] = r.walks

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.colorbar(field(axs[0], scene, X, Y, U, cmap="RdBu_r", title="u  (Laplace, Dirichlet)"),
             ax=axs[0], shrink=0.7)
im = axs[1].pcolormesh(X, Y, W.reshape(X.shape), cmap="viridis", shading="auto",
                       norm=plt.matplotlib.colors.LogNorm(), rasterized=True)
axs[1].set_aspect("equal"); axs[1].axis("off")
axs[1].set_title("walks per point to reach stderr < 0.02")
fig.colorbar(im, ax=axs[1], shrink=0.7)
save(fig, "snowflake.png")
