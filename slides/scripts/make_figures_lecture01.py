"""
Sketch-style figures for Lecture 1: Introduction to machine learning.

Run from anywhere:
    python scripts/make_figures_lecture01.py              # all figures
    python scripts/make_figures_lecture01.py house veg    # only figures whose name contains these words

Output: figures/intro-sketch/*.png (relative to the project root).
Requirements: numpy, matplotlib >= 3.7, scikit-learn (k-means and the small classifiers).
All data are synthetic, with fixed random seeds.

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
OUT = ROOT / "figures" / "intro-sketch"
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
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CLASS_COLORS = [C["red"], C["blue"], C["green"]]
CLASS_MARKERS = ["x", "o", "^"]


def class_points(ax, X, y, s=60, labels=None):
    for k in np.unique(y):
        P = X[y == k]
        lab = None if labels is None else labels[k]
        if CLASS_MARKERS[k] == "x":
            ax.scatter(P[:, 0], P[:, 1], marker="x", s=s, color=CLASS_COLORS[k], linewidths=2.2, zorder=5, label=lab)
        else:
            ax.scatter(P[:, 0], P[:, 1], marker=CLASS_MARKERS[k], s=s, facecolor="white", edgecolor=CLASS_COLORS[k],
                       linewidths=2.0, zorder=5, label=lab)


def region_plot(ax, predict, xlim, ylim, K, alpha=0.12):
    g1, g2 = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, Z, levels=np.arange(K + 1) - 0.5, colors=CLASS_COLORS[:K], alpha=alpha)
    ax.contour(g1, g2, Z, levels=np.arange(K) + 0.5, colors="k", linewidths=2.2)


def box(ax, xc, yc, text, w, h, fc="#FCE4EC", ec="k", fs=15, lw=2, color="k"):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, p, q, color="k", lw=2.0, ms=18, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=ms,
                                                    connectionstyle=f"arc3,rad={rad}"))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def plain(ax, xlabel="", ylabel="", lim=None):
    if lim:
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    despine(ax)


# --------------------------------------------------------------------------- figures
def fig_ml_types():
    rng = np.random.default_rng(1)
    A = rng.normal([-1.0, -0.7], 0.55, size=(25, 2))
    B = rng.normal([1.0, 0.8], 0.55, size=(25, 2))
    U = np.r_[rng.normal([-1.2, 1.0], 0.4, (20, 2)), rng.normal([1.2, 1.1], 0.4, (20, 2)), rng.normal([0, -1.2], 0.4, (20, 2))]
    km = KMeans(3, n_init=10, random_state=0).fit(U)
    with sketchy():
        fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2))
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.86, wspace=0.1)
        X = np.r_[A, B]
        y = np.r_[np.zeros(25, int), np.ones(25, int)]
        lr = LogisticRegression().fit(X, y)
        region_plot(axs[0], lr.predict, (-2.6, 2.6), (-2.4, 2.6), 2)
        class_points(axs[0], X, y, s=45)
        plain(axs[0], lim=((-2.6, 2.6), (-2.4, 2.6)))
        axs[0].set_title("supervised: labelled data", fontsize=16)
        axs[1].scatter(U[:, 0], U[:, 1], c=[CLASS_COLORS[k] for k in km.labels_], s=40, zorder=5)
        for cxy in km.cluster_centers_:
            axs[1].scatter(*cxy, marker="*", s=300, color="k", zorder=6)
        plain(axs[1], lim=((-2.6, 2.6), (-2.4, 2.6)))
        axs[1].set_title("unsupervised: find structure", fontsize=16)
        ax = axs[2]
        blank(ax, (0, 10), (0, 10))
        box(ax, 5, 7.6, "agent", 3.6, 1.4, fc="#E3F2FD", ec=C["blue"], fs=17)
        box(ax, 5, 2.4, "environment", 4.6, 1.4, fc="#E8F5E9", ec=C["green"], fs=17)
        arrow(ax, (7.0, 7.2), (7.0, 3.2), rad=-0.5, lw=2.2)
        ax.text(9.0, 5.2, "action", fontsize=14, ha="center")
        arrow(ax, (3.0, 3.2), (3.0, 7.2), rad=-0.5, lw=2.2)
        ax.text(1.0, 5.2, "state,\nreward", fontsize=14, ha="center")
        ax.set_title("reinforcement: learn by trial", fontsize=16)
        save(fig, "ml_types")


def fig_supervised_diagram():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11, 3.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 11), (0, 3.6))
        box(ax, 1.6, 2.7, "examples with\n\"right answers\"\n$(x, y)$", 2.8, 1.25, fc="#FFF8E1", ec=C["orange"], fs=14)
        box(ax, 5.3, 2.7, "learning\nalgorithm", 2.2, 1.1, fs=15)
        arrow(ax, (3.05, 2.7), (4.15, 2.7))
        arrow(ax, (5.3, 2.12), (5.3, 1.45))
        box(ax, 5.3, 0.95, "model $f$", 1.8, 0.85, fc="#E3F2FD", ec=C["blue"], fs=16)
        box(ax, 2.2, 0.95, "new input $x$", 2.3, 0.8, fc="#E8F5E9", ec=C["green"], fs=14)
        arrow(ax, (3.4, 0.95), (4.35, 0.95))
        arrow(ax, (6.25, 0.95), (7.3, 0.95))
        box(ax, 8.8, 0.95, "prediction $\\hat y = f(x)$", 2.9, 0.8, fc="#E8F5E9", ec=C["green"], fs=14)
        ax.text(8.6, 2.75, "training: learn $f$\nfrom the examples", fontsize=14, color=C["navy"], ha="center")
        save(fig, "supervised_diagram")


def _house_data():
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(38, 140, 24))
    y = 60 + 0.1 * x + 0.016 * x ** 2 + rng.normal(0, 12, x.size)
    return x, y


def fig_house():
    x, y = _house_data()
    lin = np.polyfit(x, y, 1)
    quad = np.polyfit(x, y, 2)
    xs = np.linspace(30, 148, 300)
    q = 86.0
    for step in (1, 2, 3):
        with sketchy():
            fig, ax = plt.subplots(figsize=(8, 4.8))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.96)
            data_points(ax, x, y, s=80)
            ax.axvline(q, color=C["grey"], ls=":", lw=1.6)
            if step > 1:
                coef = lin if step == 2 else quad
                yq = np.polyval(coef, q)
                ax.plot(xs, np.polyval(coef, xs), color=C["blue"], lw=3)
                ax.scatter([q], [yq], s=130, color=C["green"], zorder=7)
                ax.text(q - 3, yq + 22, f"86 m²: about {yq:.0f}k GBP", fontsize=15, color=C["green"], ha="right")
            ax.set_xlim(30, 148)
            ax.set_ylim(40, 400)
            ax.set_xlabel("floor area (m$^2$)")
            ax.set_ylabel("price (thousand GBP)")
            despine(ax)
            save(fig, f"house_step{step}")
    print(f"    86 m2: straight line {np.polyval(lin, q):.0f}, curve {np.polyval(quad, q):.0f}")


def _veg_data():
    rng = np.random.default_rng(7)
    carrot = np.c_[rng.normal(18, 2.0, 18), rng.normal(3.0, 0.6, 18)]
    paprika = np.c_[rng.normal(10, 1.2, 18), rng.normal(8.0, 0.9, 18)]
    tomato = np.c_[rng.normal(6.0, 0.9, 18), rng.normal(6.3, 0.8, 18)]
    X = np.r_[carrot, paprika, tomato]
    y = np.r_[np.zeros(18, int), np.ones(18, int), 2 * np.ones(18, int)]
    return X, y


def fig_vegetables():
    X, y = _veg_data()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10)).fit(X, y)
    lim = ((2, 24), (0.5, 11))
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.96)
            if step == 2:
                region_plot(ax, model.predict, *lim, 3)
            class_points(ax, X, y, s=70, labels=["carrot", "paprika", "tomato"])
            ax.set_xlim(*lim[0])
            ax.set_ylim(*lim[1])
            ax.set_xlabel("length (cm)")
            ax.set_ylabel("width (cm)")
            ax.legend(loc="upper right", fontsize=13, frameon=True)
            despine(ax)
            save(fig, f"vegetables_step{step}")


def fig_regression_vs_classification():
    x, yv = _house_data()
    X, y = _veg_data()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10)).fit(X, y)
    with sketchy():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.1, top=0.88, wspace=0.15)
        xs = np.linspace(30, 148, 200)
        a1.plot(xs, np.polyval(np.polyfit(x, yv, 2), xs), color=C["blue"], lw=3)
        data_points(a1, x, yv, s=55)
        plain(a1, "input $x$", "output $y$ (a number)", ((30, 148), (40, 400)))
        a1.set_title("regression: fit a curve", fontsize=16)
        region_plot(a2, model.predict, (2, 24), (0.5, 11), 3)
        class_points(a2, X, y, s=50)
        plain(a2, "feature $x_1$", "feature $x_2$", ((2, 24), (0.5, 11)))
        a2.set_title("classification: find decision boundaries", fontsize=16)
        save(fig, "regression_vs_classification")


def fig_venn():
    with sketchy():
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 10), (0, 6))
        ax.set_aspect("equal")
        ax.add_patch(plt.Circle((3.6, 3.0), 2.5, fc=C["blue"], alpha=0.12, ec=C["blue"], lw=2.5))
        ax.add_patch(plt.Circle((6.4, 3.0), 2.5, fc=C["red"], alpha=0.12, ec=C["red"], lw=2.5))
        ax.text(2.6, 4.3, "statistics", fontsize=19, ha="center", color=C["blue"])
        ax.text(2.4, 2.4, "models\nuncertainty\ninference", fontsize=13, ha="center", color=C["grey"])
        ax.text(7.5, 4.3, "computer\nscience", fontsize=19, ha="center", color=C["red"])
        ax.text(7.6, 2.4, "algorithms\ncomputation\ndata at scale", fontsize=13, ha="center", color=C["grey"])
        ax.text(5.0, 3.0, "machine\nlearning", fontsize=18, ha="center", va="center", color=C["purple"])
        save(fig, "venn")


def _gradient_descent(x, y, buggy, alpha=0.1, iters=5000):
    """Batch gradient descent for y ~ w x + b (Lecture 2). buggy=True drops the factor x in dJ/dw."""
    w, b = 0.0, 0.0
    for _ in range(iters):
        y_hat = w * x + b
        dw = np.mean(y_hat - y) if buggy else np.mean((y_hat - y) * x)
        db = np.mean(y_hat - y)
        w, b = w - alpha * dw, b - alpha * db
    return w, b


def fig_spot_the_bug():
    x, y = _house_data()
    x, y = x / 100, y / 100                      # floor area in 100 m^2, price in 100k GBP
    w_ok, b_ok = _gradient_descent(x, y, buggy=False)
    w_bug, b_bug = _gradient_descent(x, y, buggy=True)
    xs = np.linspace(0.3, 1.48, 100)
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.96)
        data_points(ax, x, y, s=70)
        ax.plot(xs, w_ok * xs + b_ok, color=C["blue"], lw=3, label="correct gradient")
        ax.plot(xs, w_bug * xs + b_bug, color=C["orange"], lw=3, ls="--", label="gradient missing a factor $x$")
        ax.set_xlim(0.3, 1.48)
        ax.set_xlabel("floor area (100 m$^2$)")
        ax.set_ylabel("price (100,000 GBP)")
        ax.legend(loc="upper left", fontsize=13, frameon=False)
        despine(ax)
        save(fig, "spot_the_bug")
    lsq = np.polyfit(x, y, 1)
    print(f"    correct: w = {w_ok:.3f}, b = {b_ok:.3f} (least squares: {lsq[0]:.3f}, {lsq[1]:.3f}); "
          f"buggy: w = {w_bug:.3f}, b = {b_bug:.3f}")


FIGURES = {"ml_types": fig_ml_types, "supervised_diagram": fig_supervised_diagram, "house": fig_house,
           "vegetables": fig_vegetables, "regression_vs_classification": fig_regression_vs_classification,
           "venn": fig_venn, "spot_the_bug": fig_spot_the_bug}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}  ->  {OUT}")
    for key, fn in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            print(key)
            fn()
