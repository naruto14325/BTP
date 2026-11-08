import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT, exist_ok=True)


def draw_scene(ax, scene, lw=1.0):
    scene.build()
    for a, b, c in ((scene.d_a, scene.d_b, "k"), (scene.n_a, scene.n_b, "#2a7fff")):
        for p, q in zip(a, b):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, solid_capstyle="round", zorder=4)


def field(ax, scene, X, Y, U, cmap="inferno", levels=20, title=None):
    im = ax.pcolormesh(X, Y, U, cmap=cmap, shading="auto", rasterized=True)
    ax.contour(X, Y, U, levels=levels, colors="white", linewidths=0.4, alpha=0.6)
    draw_scene(ax, scene)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return im


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.normpath(path))
