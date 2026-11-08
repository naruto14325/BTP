"""Potential flow through a channel past insulating obstacles (Walk on Stars).

Inlet/outlet are Dirichlet (u = 0 / u = 4), channel walls and obstacles are
Neumann (no flux). Uses boundary value caching: walks run only from cache
points along the boundary, and u and grad u everywhere else come from
Green's representation formula, so the velocity field is smooth.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

import wos
from wos import DIRICHLET, NEUMANN
from _plot import draw_scene, save

L, H = 4.0, 1.5
n = 100
kinds = np.repeat([NEUMANN, DIRICHLET, NEUMANN, DIRICHLET], n)
scene = wos.Scene().add_loop(wos.rectangle(0, 0, L, H, n), kinds=kinds,
                             values=lambda x, y: np.where(x > L / 2, L, 0.0))
scene.add_loop(wos.circle((1.0, 0.75), 0.3, 200), kinds=NEUMANN, hole=True)

# A rotated "airfoil-ish" ellipse and a non-convex C-shaped obstacle.
t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
e = np.stack([0.45 * np.cos(t), 0.12 * np.sin(t)], 1)
c, s = np.cos(-0.5), np.sin(-0.5)
scene.add_loop(e @ np.array([[c, s], [-s, c]]) + [2.2, 0.45], kinds=NEUMANN, hole=True)
ang = np.linspace(0.6, 2 * np.pi - 0.6, 60)
outer = np.stack([3.2 + 0.32 * np.cos(ang), 1.0 + 0.32 * np.sin(ang)], 1)
inner = np.stack([3.2 + 0.2 * np.cos(ang[::-1]), 1.0 + 0.2 * np.sin(ang[::-1])], 1)
scene.add_loop(np.vstack([outer, inner]), kinds=NEUMANN, hole=True)

t0 = time.time()
cache = wos.BoundaryCache(scene, n_points=5000, n_walks=1024, delta=0.0025)
print(f"cache: {len(cache.Y)} points, {cache.walks:,} walks in {time.time() - t0:.1f}s")
t1 = time.time()
X, Y, P, mask = scene.grid(260)
r = cache.evaluate(P[mask], gradient=True, n_walks_near=256)
print(f"evaluate: {mask.sum()} points ({r.cached.mean():.0%} from cache) in {time.time() - t1:.1f}s")

U = np.full(len(P), np.nan); U[mask] = r.value
VX = np.zeros(len(P)); VY = np.zeros(len(P))
VX[mask], VY[mask] = r.grad[:, 0], r.grad[:, 1]
ok = mask & np.isfinite(VX)
speed = np.hypot(VX, VY).reshape(X.shape)
# Fill grid cells inside obstacles with their nearest valid neighbor so the
# smooth (gouraud) shading reaches the walls; obstacles are painted on top.
idx = distance_transform_edt(~ok.reshape(X.shape), return_distances=False, return_indices=True)
speed = speed[tuple(idx)]
VX, VY = VX.reshape(X.shape)[tuple(idx)], VY.reshape(X.shape)[tuple(idx)]

fig, axs = plt.subplots(2, 1, figsize=(12, 9))
axs[0].pcolormesh(X, Y, U.reshape(X.shape), cmap="viridis", shading="auto", rasterized=True)
axs[0].contour(X, Y, U.reshape(X.shape), levels=24, colors="white", linewidths=0.5)
axs[0].set_title("potential u")
axs[1].pcolormesh(X, Y, speed, cmap="magma", shading="gouraud",
                  vmin=0, vmax=2.5 * np.median(speed), rasterized=True)
axs[1].streamplot(X, Y, VX, VY, color="white",
                  density=1.6, linewidth=0.6, arrowsize=0.6)
axs[1].set_title("speed |grad u| and streamlines")
for ax in axs:
    for pts, kinds, _ in scene._loops[1:]:
        ax.fill(pts[:, 0], pts[:, 1], color="white", zorder=3)
    draw_scene(ax, scene, lw=1.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(0, L); ax.set_ylim(0, H)
save(fig, "flow.png")
