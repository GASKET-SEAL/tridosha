"""Dosha triangle: Prakriti (fixed star), today's Vikriti, yesterday's dot, trail, 95% confidence region."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from .simplex import TRI_VERTS, to_xy

COL = {"vata": "#5B8DEF", "pitta": "#E8604C", "kapha": "#3FA66B"}


def plot_triangle(prakriti, vikriti, previous=None, trail=None, ellipse=None, title=None, path=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.add_patch(Polygon(TRI_VERTS, closed=True, fc="#FAFAF7", ec="#444", lw=1.6, zorder=1))
    for t in np.arange(0.1, 1.0, 0.1):                                    # iso-share grid
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            p1 = np.zeros(3); p1[a] = t; p1[b] = 1 - t
            p2 = np.zeros(3); p2[a] = t; p2[c] = 1 - t
            ax.plot(*zip(to_xy(p1), to_xy(p2)), color="#DDD", lw=.6, zorder=2)
    for (x, y), name, dx, dy in zip(TRI_VERTS, ("VATA", "PITTA", "KAPHA"), (-.02, .02, 0), (-.05, -.05, .035)):
        ax.text(x + dx, y + dy, name, ha="center", va="center", fontsize=13, weight="bold", color=COL[name.lower()])
    if ellipse is not None:
        ax.add_patch(Polygon(ellipse, closed=True, fc="#9467BD", alpha=.18, ec="#9467BD", lw=1, zorder=3, label="95% confidence"))
    if trail is not None and len(trail) > 1:
        xy = np.array([to_xy(t) for t in trail])
        ax.plot(xy[:, 0], xy[:, 1], "-", color="#888", lw=1, alpha=.6, zorder=4)
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color="#888", alpha=np.linspace(.15, .6, len(xy)), zorder=4)
    pxy, vxy = to_xy(prakriti), to_xy(vikriti)
    ax.annotate("", xy=vxy, xytext=pxy, arrowprops=dict(arrowstyle="->", color="#555", lw=1.3), zorder=5)
    if previous is not None:
        ax.scatter(*to_xy(previous), s=110, facecolors="none", edgecolors="#E08A00", lw=2, zorder=6, label="yesterday")
    ax.scatter(*pxy, marker="*", s=330, c="#222", zorder=7, label="Prakriti (baseline)")
    ax.scatter(*vxy, s=140, c="#9467BD", ec="white", lw=1.5, zorder=8, label="Vikriti (today)")
    ax.set_xlim(-.1, 1.1); ax.set_ylim(-.12, .98); ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0, 1.0))
    txt = "  ".join(f"{n[0].upper()} {v*100:.0f}%" for n, v in zip(("v", "p", "k"), vikriti))
    ax.set_title((title + "\n" if title else "") + "Vikriti: " + txt, fontsize=11)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fig
