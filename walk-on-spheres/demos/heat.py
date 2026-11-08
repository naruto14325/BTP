"""Steady-state heat in a room: Poisson equation with mixed boundaries.

Two heaters (Gaussian sources f), cold windows (Dirichlet u = 0), a warm
door (u = 0.5) and insulated walls everywhere else (Neumann). Solved with
boundary value caching; the source enters through a volume integral.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

import wos
from wos import DIRICHLET, NEUMANN
from _plot import field, save


@njit
def heaters(x, y):
    a = np.exp(-((x - 0.8) ** 2 + (y - 2.2) ** 2) / 0.02)
    b = np.exp(-((x - 3.0) ** 2 + (y - 0.6) ** 2) / 0.02)
    return 40.0 * (a + b)


# L-shaped room, vertices CCW. Mark windows/door per segment after densifying.
corners = np.array([[0, 0], [4, 0], [4, 1.5], [2, 1.5], [2, 3], [0, 3]], float)
pts, kinds, vals = [], [], []
for i in range(len(corners)):
    a, b = corners[i], corners[(i + 1) % len(corners)]
    for t in np.linspace(0, 1, 40, endpoint=False):
        p = a + t * (b - a)
        pts.append(p)
        window = (i == 0 and 1.0 < p[0] < 2.0) or (i == 5 and 1.2 < p[1] < 2.2)
        door = i == 2 and 2.6 < p[0] < 3.4
        kinds.append(DIRICHLET if (window or door) else NEUMANN)
        vals.append(0.5 if door else 0.0)

scene = wos.Scene().add_loop(np.array(pts), kinds=np.array(kinds), values=np.array(vals))
# An insulated pillar in the middle of the room.
scene.add_loop(wos.rectangle(0.9, 0.8, 1.3, 1.2, 4), kinds=NEUMANN, hole=True)

t0 = time.time()
cache = wos.BoundaryCache(scene, n_points=4000, n_walks=2048, delta=0.0025, source=heaters,
                          source_samples=4, volume_res=500)
X, Y, P, mask = scene.grid(300)
r = cache.evaluate(P[mask], n_walks_near=1024)
U = np.full(len(P), np.nan); U[mask] = r.value; U = U.reshape(X.shape)
print(f"{mask.sum()} points ({r.cached.mean():.0%} from cache) in {time.time() - t0:.1f}s")

fig, ax = plt.subplots(figsize=(8, 6))
fig.colorbar(field(ax, scene, X, Y, U, cmap="inferno", levels=12,
                   title="temperature (black: fixed temperature, blue: insulated)"), ax=ax)
save(fig, "heat.png")
