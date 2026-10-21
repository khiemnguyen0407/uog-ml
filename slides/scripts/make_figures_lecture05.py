"""
Sketch-style figures for Lecture 5: Support vector machines.

Run from anywhere:
    python scripts/make_figures_lecture05.py                     # all figures
    python scripts/make_figures_lecture05.py lagrange kernel     # only figures whose name contains these words

Output: figures/svm-sketch/*.png (relative to the project root).

Requirements: numpy, matplotlib >= 3.7, scikit-learn. All data are synthetic (make_blobs, make_moons)
with fixed random seeds, so the figures and the numbers quoted on the slides are reproducible.

Fonts: the hand-drawn look uses the "xkcd Script" font (scripts/fonts/xkcd-script.ttf,
CC BY-NC 3.0, https://github.com/ipython/xkcd-font). If it is missing, Matplotlib falls back to
Humor Sans / Comic Neue / Comic Sans MS, so the script still runs.

Figures that are shown one after another on the same slide ("builds") are drawn on canvases of
identical size with fixed margins (no bbox_inches="tight"), so they line up exactly.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "svm-sketch"
FONT_DIR = Path(__file__).resolve().parent / "fonts"

# Palette taken from the Beamer colour theme (beamercolorthemeCaltechBeaver.sty)
C = dict(
    navy="#000080",
    blue="#2171b5",
    red="#E53E30",
    green="#00AD69",
    purple="#652F6C",
    grey="#76777B",
    light="#C8C8C8",
    orange="#E08A00",
)
DPI = 200


# --------------------------------------------------------------------------- style
def _register_fonts() -> list[str]:
    for f in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    available = {f.name for f in font_manager.fontManager.ttflist}
    prefs = ["xkcd Script", "Humor Sans", "Comic Neue", "Comic Sans MS"]
    return [p for p in prefs if p in available] + ["DejaVu Sans"]


FONTS = _register_fonts()


@contextmanager
def sketchy():
    """xkcd-style context with the course palette and a hand-written font (also for maths)."""
    with plt.xkcd(scale=0.8, length=110, randomness=2):
        plt.rcParams.update(
            {
                "font.family": FONTS,
                "font.size": 15,
                "axes.labelsize": 16,
                "axes.titlesize": 17,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "axes.linewidth": 1.6,
                "lines.linewidth": 2.6,
                "mathtext.fontset": "custom",
                "mathtext.rm": FONTS[0],
                "mathtext.it": FONTS[0],
                "mathtext.bf": FONTS[0],
                "mathtext.fallback": "stixsans",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "axes.prop_cycle": plt.cycler(color=[C["blue"], C["red"], C["green"], C["purple"]]),
            }
        )
        yield


def despine(ax, keep=("left", "bottom")):
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(s in keep)


def arrow_axes(ax, xlabel="$x$", ylabel="$y$", xlim=(0, 10), ylim=(0, 6)):
    """Bare axes with arrow heads and no ticks (for conceptual sketches)."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    despine(ax, keep=())
    ax.set_xticks([])
    ax.set_yticks([])
    kw = dict(arrowprops=dict(arrowstyle="-|>", lw=1.8, color="k", mutation_scale=18), annotation_clip=False)
    ax.annotate("", xy=(xlim[1], ylim[0]), xytext=(xlim[0], ylim[0]), **kw)
    ax.annotate("", xy=(xlim[0], ylim[1]), xytext=(xlim[0], ylim[0]), **kw)
    ax.text(xlim[1], ylim[0] - 0.06 * (ylim[1] - ylim[0]), xlabel, ha="right", va="top", fontsize=18)
    ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), ylim[1], ylabel, ha="right", va="top", fontsize=18)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print("  wrote", name + ".png")


def data_points(ax, x, y, s=90, **kw):
    ax.scatter(x, y, marker="x", s=s, color=C["red"], linewidths=2.4, zorder=5, **kw)


# --------------------------------------------------------------------------- helpers
from sklearn.datasets import make_blobs, make_moons  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

CLASS_COLORS = [C["red"], C["blue"], C["green"], C["purple"]]
CLASS_MARKERS = ["x", "o", "^", "s"]
LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


def class_points(ax, X, y, s=70, alpha=1.0, labels=None):
    for k in np.unique(y):
        P = X[y == k]
        kw = dict(s=s, alpha=alpha, zorder=5, label=None if labels is None else labels[k])
        if CLASS_MARKERS[k] == "x":
            ax.scatter(P[:, 0], P[:, 1], marker="x", color=CLASS_COLORS[k], linewidths=2.2, **kw)
        else:
            ax.scatter(P[:, 0], P[:, 1], marker=CLASS_MARKERS[k], facecolor="white", edgecolor=CLASS_COLORS[k],
                       linewidths=2.0, **kw)


def region_plot(ax, predict, xlim, ylim, K, alpha=0.12, n=400):
    """Shade the predicted class of every point and draw the borders between regions."""
    g1, g2 = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    Z = predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, Z, levels=np.arange(K + 1) - 0.5, colors=CLASS_COLORS[:K], alpha=alpha)
    ax.contour(g1, g2, Z, levels=np.arange(K) + 0.5, colors="k", linewidths=2.2)


def plain_axes(ax, xlim, ylim, xlabel="$x_1$", ylabel="$x_2$", ticks=False):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    despine(ax)


def _separable_data():
    X, y = make_blobs(n_samples=[20, 20], centers=[[-1.3, -1.0], [1.3, 1.1]], cluster_std=0.55, random_state=6)
    return X, y


def fig_svm_margin():
    X, y = _separable_data()
    svm = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = svm.coef_[0], svm.intercept_[0]
    lim = (-3.0, 3.0)
    xs = np.array(lim)
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(5.8, 5.2))
            fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.92)
            class_points(ax, X, y, s=65)
            if step == 1:
                # other separating lines: rotate the SVM line about the centre of the gap, keep only perfect separators
                centre = 0.5 * (X[y == 0].mean(axis=0) + X[y == 1].mean(axis=0))
                ang0 = np.arctan2(-w[0], w[1])
                chosen = []
                for dang, shift in [(0.45, 0.25), (-0.5, -0.2), (0.15, -0.45), (-0.25, 0.4), (0.3, 0.0)]:
                    ang = ang0 + dang
                    d = np.array([np.cos(ang), np.sin(ang)])
                    nrm = np.array([-d[1], d[0]])
                    p0 = centre + shift * nrm
                    side = (X - p0) @ nrm
                    if np.all(side[y == 1] * np.sign(side[y == 1].mean()) > 0) and np.all(side[y == 0] * np.sign(side[y == 1].mean()) < 0):
                        chosen.append((p0, d))
                    if len(chosen) == 3:
                        break
                for (p0, d), col in zip(chosen, [C["grey"], C["purple"], C["orange"]]):
                    t = np.linspace(-6, 6, 2)
                    ax.plot(p0[0] + t * d[0], p0[1] + t * d[1], color=col, lw=2.4)
                ax.set_title("many lines separate the data", fontsize=16)
            else:
                for off, ls in [(0, "-"), (1, "--"), (-1, "--")]:
                    ax.plot(xs, -(w[0] * xs + b - off) / w[1], color="k", lw=2.6 if off == 0 else 1.8, ls=ls)
                sv = svm.support_vectors_
                ax.scatter(sv[:, 0], sv[:, 1], s=320, facecolor="none", edgecolor=C["navy"], linewidths=2.2, zorder=6)
                ax.set_title("SVM: the largest margin", fontsize=16)
                ax.text(0.45, -1.95, "solid: $\\mathbf{w}\\cdot\\mathbf{x} + b = 0$", fontsize=13, bbox=LABEL_BOX)
                ax.text(0.45, -2.35, "dashed: $\\mathbf{w}\\cdot\\mathbf{x} + b = \\pm 1$", fontsize=13, bbox=LABEL_BOX)
                ax.text(0.45, -2.75, "circled: support vectors", fontsize=13, color=C["navy"], bbox=LABEL_BOX)
            plain_axes(ax, lim, lim)
            ax.set_aspect("equal")
            save(fig, f"svm_margin_step{step}")
    print(f"    linear SVM: margin width 2/|w| = {2 / np.linalg.norm(w):.3f}, support vectors {len(svm.support_vectors_)}")


def fig_svm_soft():
    X, y = make_blobs(n_samples=[40, 40], centers=[[-0.9, -0.7], [0.9, 0.8]], cluster_std=0.85, random_state=3)
    lim = (-3.2, 3.2)
    xs = np.array(lim)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.9))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.88, wspace=0.12)
        for ax, Cv in zip(axs, [0.01, 100.0]):
            svm = SVC(kernel="linear", C=Cv).fit(X, y)
            w, b = svm.coef_[0], svm.intercept_[0]
            class_points(ax, X, y, s=45)
            for off, ls in [(0, "-"), (1, "--"), (-1, "--")]:
                ax.plot(xs, -(w[0] * xs + b - off) / w[1], color="k", lw=2.6 if off == 0 else 1.8, ls=ls)
            sv = svm.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolor="none", edgecolor=C["navy"], linewidths=1.6, zorder=6)
            ax.set_title(f"$C = {Cv:g}$: {'wide' if Cv < 1 else 'narrow'} margin, {len(sv)} support vectors",
                         fontsize=15)
            plain_axes(ax, lim, lim)
            ax.set_aspect("equal")
        save(fig, "svm_soft_margin")


def fig_svm_kernel():
    Xm, ym = make_moons(200, noise=0.2, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)
    lim_x, lim_y = (-1.6, 2.6), (-1.2, 1.7)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.88, wspace=0.12)
        for ax, kern in zip(axs, ["linear", "rbf"]):
            svm = SVC(kernel=kern, C=1.0).fit(X_tr, y_tr)
            region_plot(ax, svm.predict, lim_x, lim_y, 2)
            class_points(ax, X_tr, y_tr, s=40)
            sv = svm.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolor="none", edgecolor=C["navy"], linewidths=1.2, zorder=6)
            name = "linear kernel" if kern == "linear" else "Gaussian (RBF) kernel"
            ax.set_title(f"{name}: test accuracy {svm.score(X_te, y_te):.1%}", fontsize=15)
            plain_axes(ax, lim_x, lim_y)
        save(fig, "svm_kernel")


def _lines(ax, w, b, xs, lw=2.6):
    """Decision line w.x + b = 0 (solid) and the margins w.x + b = +-1 (dashed)."""
    for off, ls in [(0, "-"), (1, "--"), (-1, "--")]:
        ax.plot(xs, -(w[0] * xs + b - off) / w[1], color="k", lw=lw if off == 0 else 1.8, ls=ls)


def fig_geometry():
    """Hyperplane, normal vector w, and the distance of a point to the hyperplane."""
    w = np.array([1.0, 2.0])
    b = -4.0
    x0 = np.array([3.4, 2.6])
    nw = np.linalg.norm(w)
    proj = x0 - (w @ x0 + b) / nw ** 2 * w                 # foot of the perpendicular
    xs = np.array([-0.5, 5.0])
    with sketchy():
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        fig.subplots_adjust(left=0.06, right=0.97, bottom=0.06, top=0.97)
        ax.plot(xs, -(w[0] * xs + b) / w[1], color="k", lw=2.6)
        ax.text(4.2, -(w[0] * 4.2 + b) / w[1] - 0.45, "$\\mathbf{w}\\cdot\\mathbf{x} + b = 0$", fontsize=15)
        p = np.array([1.2, -(w[0] * 1.2 + b) / w[1]])
        ax.annotate("", xy=p + 0.55 * w, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=2.4, color=C["blue"],
                                                                 mutation_scale=18))
        ax.text(*(p + 0.55 * w + np.array([0.05, 0.1])), "$\\mathbf{w}$", fontsize=17, color=C["blue"])
        ax.plot([x0[0], proj[0]], [x0[1], proj[1]], color=C["red"], lw=2.2, ls="--")
        ax.scatter(*x0, s=90, color=C["red"], zorder=5)
        ax.text(x0[0] + 0.12, x0[1] + 0.08, "$\\mathbf{x}$", fontsize=17, color=C["red"])
        mid = 0.5 * (x0 + proj)
        ax.text(mid[0] + 0.15, mid[1] - 0.05,
                "distance $= \\dfrac{|\\mathbf{w}\\cdot\\mathbf{x} + b|}{\\|\\mathbf{w}\\|}$", fontsize=15, color=C["red"])
        ax.text(0.0, 3.45, "$\\mathbf{w}\\cdot\\mathbf{x} + b > 0$", fontsize=14, color=C["blue"])
        ax.text(0.0, 0.2, "$\\mathbf{w}\\cdot\\mathbf{x} + b < 0$", fontsize=14, color=C["red"])
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.2, 4.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)
        save(fig, "geometry")


def fig_lagrange_equality():
    """min x1^2 + x2^2 subject to x1 + x2 = 1: the optimum is where a contour touches the line."""
    g1, g2 = np.meshgrid(np.linspace(-0.6, 1.6, 300), np.linspace(-0.6, 1.6, 300))
    F = g1 ** 2 + g2 ** 2
    with sketchy():
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.11, top=0.97)
        cs = ax.contour(g1, g2, F, levels=[0.05, 0.15, 0.3, 0.5, 0.8, 1.2], colors=C["grey"], linewidths=1.3)
        ax.clabel(cs, fontsize=9, fmt="%.2g")
        xs = np.array([-0.6, 1.6])
        ax.plot(xs, 1 - xs, color=C["blue"], lw=2.8)
        ax.text(1.05, -0.45, "$g(\\mathbf{x}) = x_1 + x_2 - 1 = 0$", fontsize=13, color=C["blue"], ha="center")
        opt = np.array([0.5, 0.5])
        ax.scatter(*opt, s=110, color=C["red"], zorder=6)
        ax.annotate("", xy=opt + 0.35 * np.array([1, 1]), xytext=opt,
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=C["red"], mutation_scale=16))
        ax.annotate("", xy=opt + 0.2 * np.array([1, 1]) + np.array([0.06, -0.06]), xytext=opt + np.array([0.06, -0.06]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=C["blue"], mutation_scale=16))
        ax.text(0.93, 0.9, "$\\nabla f$", fontsize=15, color=C["red"])
        ax.text(0.8, 0.5, "$\\nabla g$", fontsize=15, color=C["blue"])
        ax.text(-0.5, 1.4, "contours of $f = x_1^2 + x_2^2$", fontsize=12, color=C["grey"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        despine(ax)
        save(fig, "lagrange_equality")


def fig_lagrange_inequality():
    """min (x - 2)^2 subject to x <= 3 (inactive) and x <= 1 (active)."""
    xs = np.linspace(-0.5, 4.5, 300)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.16, top=0.86, wspace=0.12)
        for ax, bound, title in [(axs[0], 3.0, "$x \\leq 3$: constraint inactive, $\\mu = 0$"),
                                 (axs[1], 1.0, "$x \\leq 1$: constraint active, $\\mu = 2 > 0$")]:
            ax.axvspan(-0.5, bound, color=C["green"], alpha=0.10)
            ax.axvline(bound, color=C["green"], lw=2)
            ax.plot(xs, (xs - 2) ** 2, color="k", lw=2.6)
            xopt = min(2.0, bound)
            ax.scatter([xopt], [(xopt - 2) ** 2], s=110, color=C["red"], zorder=6)
            ax.text(bound - 0.1, 5.4, "feasible", ha="right", fontsize=13, color=C["green"])
            ax.text(xopt, (xopt - 2) ** 2 + 0.55, f"$x^* = {xopt:g}$", ha="center", fontsize=14, color=C["red"])
            ax.set_title(title, fontsize=15)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.4, 6.2)
            ax.set_xlabel("$x$")
            ax.set_yticks([])
            despine(ax)
        axs[0].set_ylabel("$f(x) = (x - 2)^2$")
        save(fig, "lagrange_inequality")


WORKED_X = np.array([[2.0, 2.0], [3.0, 3.0], [0.0, 0.0]])
WORKED_Y = np.array([1, 1, -1])


def fig_worked_example():
    svm = SVC(kernel="linear", C=1e6).fit(WORKED_X, WORKED_Y)
    w, b = svm.coef_[0], svm.intercept_[0]
    alpha = np.zeros(3)
    alpha[svm.support_] = np.abs(svm.dual_coef_[0])
    xs = np.array([-1.0, 4.0])
    with sketchy():
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)
        _lines(ax, w, b, xs)
        cls = (WORKED_Y > 0).astype(int)
        class_points(ax, WORKED_X, cls, s=120)
        ax.scatter(*svm.support_vectors_.T, s=420, facecolor="none", edgecolor=C["navy"], linewidths=2.2, zorder=6)
        names = ["$\\mathbf{x}^{(1)} = (2, 2)$", "$\\mathbf{x}^{(2)} = (3, 3)$", "$\\mathbf{x}^{(3)} = (0, 0)$"]
        offs = [(0.25, -0.35), (0.25, -0.35), (0.25, -0.35)]
        for k in range(3):
            ax.text(WORKED_X[k, 0] + offs[k][0], WORKED_X[k, 1] + offs[k][1],
                    f"{names[k]}\n$\\alpha_{k + 1} = {alpha[k]:g}$", fontsize=12, bbox=LABEL_BOX)
        ax.set_xlim(-1.0, 4.0)
        ax.set_ylim(-1.0, 4.0)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        despine(ax)
        save(fig, "worked_example")
    print(f"    worked example: w = {w}, b = {b:.3f}, alpha = {alpha}, margin = {2 / np.linalg.norm(w):.4f}")


def fig_soft_alphas():
    """Soft margin, C = 1: points coloured by their multiplier alpha_i."""
    X, y = make_blobs(n_samples=[40, 40], centers=[[-0.9, -0.7], [0.9, 0.8]], cluster_std=0.85, random_state=3)
    Cv = 1.0
    svm = SVC(kernel="linear", C=Cv).fit(X, y)
    w, b = svm.coef_[0], svm.intercept_[0]
    alpha = np.zeros(len(y))
    alpha[svm.support_] = np.abs(svm.dual_coef_[0])
    zero = alpha < 1e-8
    bound = alpha >= Cv - 1e-8
    free = ~zero & ~bound
    lim = (-3.2, 3.2)
    with sketchy():
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.97)
        _lines(ax, w, b, np.array(lim))
        for mask, col, lab in [(zero, C["light"], f"$\\alpha_i = 0$: outside the margin ({zero.sum()})"),
                               (free, C["blue"], f"$0 < \\alpha_i < C$: on the margin ({free.sum()})"),
                               (bound, C["red"], f"$\\alpha_i = C$: inside or wrong side ({bound.sum()})")]:
            ax.scatter(X[mask, 0], X[mask, 1], s=60 if col != C["light"] else 40, color=col, edgecolor="k",
                       linewidths=0.6, zorder=5, label=lab)
        ax.legend(loc="upper left", fontsize=12, frameon=True)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)
        save(fig, "soft_alphas")
    yy = 2 * y - 1
    f = X @ w + b
    print(f"    soft margin C=1: alpha=0 {zero.sum()}, free {free.sum()}, at C {bound.sum()}, "
          f"misclassified {(yy * f < 0).sum()}")


def fig_hinge_vs_logistic():
    t = np.linspace(-2.5, 3.0, 400)                       # t = y f(x)
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.15, top=0.96)
        ax.plot(t, (t < 0).astype(float), color=C["grey"], lw=2.2, ls=":", label="0-1 loss (misclassified or not)")
        ax.plot(t, np.maximum(0, 1 - t), color=C["red"], lw=3, label="hinge loss $\\max(0, 1 - t)$ (SVM)")
        ax.plot(t, np.log1p(np.exp(-t)) / np.log(2), color=C["blue"], lw=3,
                label="logistic loss $\\log_2(1 + e^{-t})$ (Lecture 3)")
        ax.axvline(1, color=C["light"], lw=1.2)
        ax.text(1.08, 1.45, "margin\n$t = 1$", fontsize=12, color=C["grey"])
        ax.set_xlim(-2.5, 3.0)
        ax.set_ylim(-0.1, 3.6)
        ax.set_xlabel("$t = y\\,f(\\mathbf{x})$, with $y \\in \\{-1, +1\\}$")
        ax.set_ylabel("loss")
        ax.legend(loc="upper right", fontsize=12, frameon=False)
        despine(ax)
        save(fig, "hinge_vs_logistic")


def fig_rbf_grid():
    Xm, ym = make_moons(200, noise=0.2, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)
    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)
    lim_x, lim_y = (-2.3, 2.5), (-2.4, 2.4)
    with sketchy():
        fig, axs = plt.subplots(2, 3, figsize=(12, 7.2))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.03, top=0.93, wspace=0.08, hspace=0.22)
        for r, Cv in enumerate([0.1, 100]):
            for c, g in enumerate([0.1, 1, 10]):
                ax = axs[r, c]
                svm = SVC(kernel="rbf", C=Cv, gamma=g).fit(Xs_tr, y_tr)
                region_plot(ax, svm.predict, lim_x, lim_y, 2, n=250)
                class_points(ax, Xs_tr, y_tr, s=22)
                ax.set_title(f"$C = {Cv:g}$, $\\gamma = {g:g}$: test {svm.score(Xs_te, y_te):.0%}", fontsize=13)
                plain_axes(ax, lim_x, lim_y, xlabel="", ylabel="")
        save(fig, "rbf_grid")


FIGURES = {
    "svm_margin": fig_svm_margin,
    "geometry": fig_geometry,
    "lagrange_equality": fig_lagrange_equality,
    "lagrange_inequality": fig_lagrange_inequality,
    "worked_example": fig_worked_example,
    "svm_soft": fig_svm_soft,
    "soft_alphas": fig_soft_alphas,
    "hinge_vs_logistic": fig_hinge_vs_logistic,
    "svm_kernel": fig_svm_kernel,
    "rbf_grid": fig_rbf_grid,
}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}  ->  {OUT}")
    for key, fn in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            print(key)
            fn()
