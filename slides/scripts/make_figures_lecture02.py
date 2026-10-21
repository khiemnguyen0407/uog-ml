"""
Sketch-style figures for Lecture 2: Linear regression.
    Part 2a, one feature   -> figures/linear-regression-sketch/
    Part 2b, many features -> figures/multi-linear-regression-sketch/

Run from anywhere:
    python scripts/make_figures_lecture02.py                 # all figures
    python scripts/make_figures_lecture02.py cost scaling    # only figures whose name contains these words

Data:
    data/diabetes_bmi.csv   BMI and disease progression of 442 patients (part 2a)
    data/diabetes.csv       all 10 features of the same patients (part 2b)
Both come from scikit-learn's diabetes data set (load_diabetes(scaled=False)).

Requirements: numpy, matplotlib >= 3.7, scikit-learn.

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
OUT_A = ROOT / "figures" / "linear-regression-sketch"         # part 2a
OUT_B = ROOT / "figures" / "multi-linear-regression-sketch"   # part 2b
OUT = OUT_A                                                   # set per figure in __main__
DATA_BMI = ROOT / "data" / "diabetes_bmi.csv"
DATA_ALL = ROOT / "data" / "diabetes.csv"
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


# =========================================================================== part 2a
# --------------------------------------------------------------------------- data
def load_bmi():
    x, y = np.loadtxt(DATA_BMI, delimiter=",", skiprows=1, unpack=True)
    return x, y


def fit_line(x, y):
    """Least-squares line (closed form for univariate linear regression)."""
    xb, yb = x.mean(), y.mean()
    w = ((x - xb) * (y - yb)).sum() / ((x - xb) ** 2).sum()
    return w, yb - w * xb


TOY_X = np.array([1.0, 2.0, 3.0])
TOY_Y = TOY_X.copy()


def toy_cost(w, b=0.0):
    w = np.asarray(w, dtype=float)
    return sum((w * xi + b - yi) ** 2 for xi, yi in zip(TOY_X, TOY_Y)) / (2 * len(TOY_X))


# --------------------------------------------------------------------------- figures
def fig_bmi():
    """Motivating data set: 442 patients, BMI vs disease progression (two-step build)."""
    x, y = load_bmi()
    w, b = fit_line(x, y)
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(8, 5.4))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.96)
            ax.scatter(x, y, marker="x", s=26, color=C["red"], linewidths=1.3, alpha=0.75)
            ax.set_xlabel("body mass index, BMI ($\\mathrm{kg/m^2}$)")
            ax.set_ylabel("disease progression after 1 year")
            ax.set_xlim(16, 44)
            ax.set_ylim(0, 370)
            despine(ax)
            if step == 2:
                xs = np.array([17, 43])
                ax.plot(xs, w * xs + b, color=C["blue"], lw=3.2)
                ax.text(17.2, 330, f"$\\hat{{y}} = {w:.2f}\\,x - {abs(b):.2f}$", color=C["blue"], fontsize=19)
            save(fig, f"bmi_data_step{step}")


def fig_pipeline():
    """Training set -> learning algorithm -> model f;  x -> f -> y_hat."""
    with sketchy():
        fig, ax = plt.subplots(figsize=(10, 4.6))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.6)
        ax.axis("off")

        def box(xc, yc, text, w=2.6, h=0.75, fc="#FCE4EC", ec="k", fs=17, color="k"):
            ax.add_patch(
                FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                               fc=fc, ec=ec, lw=2)
            )
            ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color)

        def arrow(p, q):
            ax.annotate("", xy=q, xytext=p,
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="k", mutation_scale=20))

        box(5, 4.0, "training set")
        box(5, 2.6, "learning algorithm")
        box(5, 1.0, "model  $f$", w=1.9)
        box(1.5, 1.0, "$x$\n(feature)", w=2.0, h=1.15, fc="#FFF8E1", ec=C["orange"])
        box(8.5, 1.0, "$\\hat{y} = f(x)$\n(prediction)", w=2.3, h=1.15, fc="#FFF8E1", ec=C["orange"])
        arrow((5, 3.6), (5, 3.0))
        arrow((5, 2.2), (5, 1.42))
        arrow((2.55, 1.0), (4.0, 1.0))
        arrow((6.0, 1.0), (7.3, 1.0))
        save(fig, "pipeline")


def fig_slope_intercept():
    """Meaning of w (slope) and b (intercept)."""
    with sketchy():
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.97)
        arrow_axes(ax, xlim=(0, 6), ylim=(0, 5.2))
        w, b = 0.55, 1.2
        xs = np.array([0, 5.8])
        ax.plot(xs, w * xs + b, color=C["blue"], lw=3)
        ax.text(3.9, w * 5.6 + b - 0.05, "$f(x) = wx + b$", color=C["blue"], fontsize=18, ha="center", va="bottom")
        # intercept
        ax.scatter([0], [b], s=110, color=C["green"], zorder=6, clip_on=False)
        ax.annotate("$b$: value at $x = 0$", xy=(0.05, b), xytext=(0.6, 0.35), fontsize=16, color=C["green"],
                    arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.8))
        # slope triangle
        x0, x1 = 2.6, 3.6
        ax.plot([x0, x1], [w * x0 + b] * 2, color=C["purple"], lw=2)
        ax.plot([x1, x1], [w * x0 + b, w * x1 + b], color=C["purple"], lw=2)
        ax.text((x0 + x1) / 2, w * x0 + b - 0.12, "1", ha="center", va="top", fontsize=17, color=C["purple"])
        ax.text(x1 + 0.1, w * (x0 + x1) / 2 + b, "$w$", ha="left", va="center", fontsize=19, color=C["purple"])
        ax.text(4.0, 1.25, "$x$ up by 1\n$\\Rightarrow$ $f(x)$ up by $w$", fontsize=15, color=C["purple"])
        save(fig, "slope_intercept")


def fig_residuals_two_points():
    """Vertical distances between two points and a line; distances depend on (w, b). Three-step build."""
    p = np.array([[2.2, 1.6], [7.2, 2.9]])
    lines = [(0.42, 1.95, C["blue"]), (0.05, 3.3, C["grey"])]
    for step in (1, 2, 3):
        with sketchy():
            fig, ax = plt.subplots(figsize=(8, 4.8))
            fig.subplots_adjust(left=0.06, right=0.97, bottom=0.10, top=0.97)
            arrow_axes(ax, xlim=(0, 10), ylim=(0, 6))
            data_points(ax, p[:, 0], p[:, 1], s=140)
            if step >= 2:
                w, b, col = lines[0]
                xs = np.array([0.8, 9.2])
                ax.plot(xs, w * xs + b, color=col, lw=3)
                ax.text(4.4, w * 4.4 + b + 0.35, "$f_{w,b}(x) = wx + b$", color=col, fontsize=18,
                        rotation=np.degrees(np.arctan(w * 4.8 / 6 * 10 / 8 * 0.62)), ha="center")
                for k, (xi, yi) in enumerate(p, start=1):
                    yl = w * xi + b
                    ax.plot([xi, xi], [yi, yl], ls="--", color=C["purple"], lw=2.2)
                    ax.text(xi + 0.18, (yi + yl) / 2, f"$d_{k}(w,b)$", fontsize=17, color=C["purple"], va="center")
            if step == 3:
                w, b, col = lines[1]
                xs = np.array([0.8, 9.2])
                ax.plot(xs, w * xs + b, color=col, lw=2.4, ls=(0, (6, 4)))
                ax.text(0.9, w * 0.8 + b + 0.2, "another choice of $(w, b)$", color=col, fontsize=15, va="bottom")
                for xi, yi in p:
                    yl = w * xi + b
                    ax.plot([xi - 0.12, xi - 0.12], [yi, yl], ls=":", color=col, lw=2.2)
            save(fig, f"residuals_two_points_step{step}")


def fig_residuals_many():
    """Many points; then a line with all vertical distances. Two-step build."""
    rng = np.random.default_rng(7)
    x = np.linspace(1.2, 8.8, 12) + rng.uniform(-0.25, 0.25, 12)
    y = 0.45 * x + 1.0 + rng.normal(0, 0.45, x.size)
    w, b = fit_line(x, y)
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(8, 4.8))
            fig.subplots_adjust(left=0.06, right=0.97, bottom=0.10, top=0.97)
            arrow_axes(ax, xlim=(0, 10), ylim=(0, 6))
            data_points(ax, x, y, s=110)
            if step == 2:
                xs = np.array([0.5, 9.5])
                ax.plot(xs, w * xs + b, color=C["blue"], lw=3)
                for xi, yi in zip(x, y):
                    ax.plot([xi, xi], [yi, w * xi + b], ls="--", color=C["purple"], lw=2)
                ax.text(x[0] - 0.15, (y[0] + w * x[0] + b) / 2, "$d_1$", fontsize=17, color=C["purple"],
                        ha="right", va="center")
                ax.text(x[-1] + 0.15, (y[-1] + w * x[-1] + b) / 2, "$d_m$", fontsize=17, color=C["purple"],
                        va="center")
            save(fig, f"residuals_many_step{step}")


def fig_abs_vs_squared():
    """Cross-sections b = 2 of G (absolute) and Q (squared) for points (1,3), (2,4), (3,5)."""
    w = np.linspace(-0.5, 2.5, 400)
    G = 2 * np.abs(w - 1)
    Q = 14 / 3 * (w - 1) ** 2
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.88, wspace=0.28)
        for ax, f, col, title in [
            (axs[0], G, C["purple"], "absolute: $G(w, 2) = 2\\,|w - 1|$"),
            (axs[1], Q, C["blue"], "squared: $Q(w, 2) = \\frac{14}{3}(w - 1)^2$"),
        ]:
            ax.plot(w, f, color=col, lw=3)
            ax.set_title(title, color=col)
            ax.set_xlabel("$w$")
            ax.set_ylim(-0.2, 5.5)
            ax.set_xlim(-0.5, 2.5)
            despine(ax)
            ax.scatter([1], [0], s=110, color=C["red"], zorder=6)
        axs[0].annotate("kink: no derivative\nat the minimum", xy=(1, 0.05), xytext=(1.25, 2.6), fontsize=15,
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="k"))
        axs[1].annotate("smooth: derivative\nis 0 at the minimum", xy=(1, 0.05), xytext=(1.25, 2.6), fontsize=15,
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="k"))
        axs[1].plot([0.55, 1.45], [0, 0], color=C["green"], lw=2.5)
        save(fig, "abs_vs_squared")


def fig_toy_cost_build():
    """b = 0: fit for w in {1, 0.5, 0} and the corresponding values of J(w); then the full parabola."""
    ws = [1.0, 0.5, 0.0]
    wgrid = np.linspace(-0.5, 2.5, 300)
    shades = [C["blue"], "#4C93C9", "#8DB9DD"]
    for step in (1, 2, 3, 4):
        with sketchy():
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1))
            fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.95, wspace=0.28)
            # left: data and model
            xs = np.array([0, 3.3])
            if step <= 3:
                w = ws[step - 1]
                a1.plot(xs, w * xs, color=C["blue"], lw=3)
                for xi, yi in zip(TOY_X, TOY_Y):
                    if abs(w * xi - yi) > 1e-9:
                        a1.plot([xi, xi], [yi, w * xi], ls="--", color=C["purple"], lw=2.2)
                a1.text(0.15, 2.85, f"$w = {w:g}$\n$f_w(x) = {'x' if w == 1 else ('0' if w == 0 else str(w) + 'x')}$",
                        fontsize=17, va="top", color=C["blue"])
            else:
                for w, col in zip(ws, shades):
                    a1.plot(np.array([0, 3.2]), w * np.array([0, 3.2]), color=col, lw=2.6)
                    a1.text(3.28, w * 3.2, f"$w = {w:g}$", color=col, fontsize=14, va="center")
            data_points(a1, TOY_X, TOY_Y, s=120)
            a1.set_xlim(0, 3.9)
            a1.set_ylim(-0.1, 3.5)
            a1.set_xticks([0, 1, 2, 3])
            a1.set_yticks([0, 1, 2, 3])
            a1.set_xlabel("$x$")
            a1.set_ylabel("$y$")
            despine(a1)
            # right: cost
            if step == 4:
                a2.plot(wgrid, toy_cost(wgrid), color=C["grey"], lw=2.6, zorder=2)
                a2.text(1.0, 5.2, "$J(w) = \\frac{7}{3}(w-1)^2$", fontsize=17, color=C["grey"], ha="center")
            shown = ws[: min(step, 3)]
            a2.scatter(shown, toy_cost(np.array(shown)), marker="x", s=130, color=C["red"], linewidths=2.6, zorder=5)
            if step <= 3:
                w = ws[step - 1]
                a2.annotate(f"$J({w:g}) = {toy_cost(w):.3g}$", xy=(w, toy_cost(w)), xytext=(w + 0.35, toy_cost(w) + 1.4),
                            fontsize=16, arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k"))
            a2.set_xlim(-0.5, 2.5)
            a2.set_ylim(-0.2, 6)
            a2.set_xlabel("$w$")
            a2.set_ylabel("$J(w)$")
            despine(a2)
            save(fig, f"toy_cost_step{step}")


def _surface_and_contour(fname, f, xr, yr, labels, levels, minimum, zticks=None, view=(28, -55)):
    X, Y = np.meshgrid(np.linspace(*xr, 160), np.linspace(*yr, 160))
    Z = f(X, Y)
    with sketchy():
        fig = plt.figure(figsize=(11, 4.6))
        ax3 = fig.add_axes([0.0, 0.02, 0.5, 0.96], projection="3d")
        ax3.plot_surface(X, Y, Z, cmap="viridis", rstride=8, cstride=8, edgecolor="k", lw=0.35, alpha=0.9)
        ax3.view_init(*view)
        ax3.set_xlabel(labels[0], labelpad=6)
        ax3.set_ylabel(labels[1], labelpad=6)
        ax3.set_zlabel(labels[2], labelpad=6)
        ax3.locator_params(nbins=4)
        if zticks is not None:
            ax3.set_zticks(zticks)
        ax3.xaxis.pane.set_alpha(0.0)
        ax3.yaxis.pane.set_alpha(0.0)
        ax3.zaxis.pane.set_alpha(0.0)
        ax = fig.add_axes([0.58, 0.14, 0.38, 0.8])
        cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=2)
        ax.clabel(cs, fontsize=11, fmt="%g")
        ax.scatter(*minimum, marker="*", s=260, color=C["red"], zorder=6)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        despine(ax)
        save(fig, fname)


def fig_bowl():
    _surface_and_contour(
        "bowl_surface_contour", lambda a, b: a ** 2 + b ** 2, (-1.2, 1.2), (-1.2, 1.2),
        ("$x_1$", "$x_2$", "$f$"), [0.05, 0.2, 0.45, 0.8, 1.25], (0, 0), zticks=[0, 1, 2],
    )


def fig_toy_cost_surface():
    f = lambda w, b: 7 / 3 * (w - 1) ** 2 + 2 * (w - 1) * b + b ** 2 / 2  # noqa: E731
    _surface_and_contour(
        "toy_cost_surface_contour", f, (-2, 4), (-8, 8), ("$w$", "$b$", "$J$"),
        [0.5, 2, 5, 10, 20, 35, 55], (1, 0), zticks=[0, 50, 100], view=(30, -125),
    )


def fig_stationary_1d():
    f = lambda x: x ** 3 + 6 * x ** 2 + 8 * x  # noqa: E731
    xs = np.linspace(-4.55, 0.45, 300)
    xmax, xmin = -2 - 2 / np.sqrt(3), -2 + 2 / np.sqrt(3)
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(8, 4.6))
            fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.95)
            ax.plot(xs, f(xs), color=C["blue"], lw=3)
            ax.axhline(0, color="k", lw=1.2)
            ax.set_xlim(-4.6, 0.6)
            ax.set_ylim(-6.5, 5)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            despine(ax)
            ax.text(-2.35, 4.0, "$f(x) = x^3 + 6x^2 + 8x$", fontsize=17, color=C["blue"])
            if step == 2:
                for xc, lab, dy in [(xmax, "local max\n$f'(x) = 0$", 0.6), (xmin, "local min\n$f'(x) = 0$", -2.6)]:
                    ax.plot([xc - 0.9, xc + 0.9], [f(xc)] * 2, ls="--", color="k", lw=2)
                    ax.scatter([xc], [f(xc)], s=130, marker="x", color=C["red"], linewidths=3, zorder=6)
                    ax.text(xc, f(xc) + dy, lab, ha="center", fontsize=15)
            save(fig, f"stationary_1d_step{step}")


def fig_stationary_2d():
    X, Y = np.meshgrid(np.linspace(-1.1, 1.1, 120), np.linspace(-1.1, 1.1, 120))
    with sketchy():
        fig = plt.figure(figsize=(6.4, 5.0))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
        ax.plot_surface(X, Y, X ** 2 + Y ** 2, cmap="viridis", rstride=8, cstride=8, edgecolor="k", lw=0.35,
                        alpha=0.85)
        ax.plot_surface(X, Y, np.zeros_like(X), color=C["purple"], alpha=0.35, rstride=60, cstride=60, lw=0)
        ax.scatter([0], [0], [0], s=90, color=C["red"], depthshade=False)
        fig.text(0.04, 0.9, "tangent plane at the\nminimum is horizontal", fontsize=15, color=C["purple"], va="top")
        ax.view_init(24, -58)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zticks([0, 1, 2])
        ax.locator_params(axis="x", nbins=3)
        ax.locator_params(axis="y", nbins=3)
        for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            p.set_alpha(0.0)
        save(fig, "stationary_2d")


def _landscape(x, y):
    """Two valleys of different depth plus a gentle bowl."""
    return (-1.0 * np.exp(-((x - 1.0) ** 2 + (y + 0.3) ** 2) / 0.7)
            - 0.75 * np.exp(-((x + 1.1) ** 2 + (y - 0.6) ** 2) / 0.5)
            + 0.08 * (x ** 2 + y ** 2))


def _grad(f, p, h=1e-5):
    x, y = p
    return np.array([(f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h)])


def _descend(f, p0, alpha, n):
    path = [np.array(p0, float)]
    for _ in range(n):
        path.append(path[-1] - alpha * _grad(f, path[-1]))
    return np.array(path)


def fig_gd_landscape():
    """Gradient descent from two nearby starts ends in two different minima."""
    X, Y = np.meshgrid(np.linspace(-2.4, 2.4, 160), np.linspace(-2.0, 2.2, 160))
    Z = _landscape(X, Y)
    starts = [(-0.05, 1.55), (0.35, 1.6)]
    paths = [_descend(_landscape, s, 0.12, 120) for s in starts]
    cols = [C["red"], C["orange"]]
    with sketchy():
        fig = plt.figure(figsize=(11, 4.6))
        ax3 = fig.add_axes([0.0, 0.02, 0.5, 0.96], projection="3d")
        ax3.computed_zorder = False  # draw the paths on top of the surface
        ax3.plot_surface(X, Y, Z, cmap="viridis", rstride=8, cstride=8, edgecolor="k", lw=0.3, alpha=0.75,
                         zorder=1)
        for p, col in zip(paths, cols):
            ax3.plot(p[:, 0], p[:, 1], _landscape(p[:, 0], p[:, 1]) + 0.02, color=col, lw=3, zorder=10)
            ax3.scatter(*p[0], _landscape(*p[0]) + 0.02, color=col, s=60, depthshade=False)
        ax3.view_init(38, -62)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.set_xlabel("$x_1$")
        ax3.set_ylabel("$x_2$")
        for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
            pane.set_alpha(0.0)
        ax = fig.add_axes([0.58, 0.14, 0.38, 0.8])
        ax.contour(X, Y, Z, levels=14, cmap="viridis", linewidths=1.6)
        for p, col, lab, ha, dx in zip(paths, cols, ["start A", "start B"], ["right", "left"], [-0.12, 0.12]):
            ax.plot(p[:, 0], p[:, 1], color=col, lw=2.8)
            ax.scatter(*p[0], color=col, s=70, zorder=6)
            ax.scatter(*p[-1], color=col, marker="*", s=240, zorder=6)
            ax.text(p[0, 0] + dx, p[0, 1] + 0.12, lab, color=col, ha=ha, fontsize=14)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)
        save(fig, "gd_landscape")


def fig_gradient_direction():
    """Contours of x1^2 + x2^2, a point a, unit directions, tangent, and +/- gradient. Two-step build."""
    a = np.array([0.72, 0.54])  # f(a) = 0.81
    g = 2 * a
    n = g / np.linalg.norm(g)
    t = np.array([-n[1], n[0]])
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
    levels = [0.1, 0.35, 0.81, 1.5]
    label_at = [(-r / np.sqrt(2), -r / np.sqrt(2)) for r in np.sqrt(levels)]
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(6.2, 5.6))
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.12, top=0.97)
            cs = ax.contour(X, Y, X ** 2 + Y ** 2, levels=levels, cmap="viridis", linewidths=2)
            ax.clabel(cs, fontsize=11, fmt="%g", manual=label_at)
            ax.scatter(*a, s=110, color=C["red"], zorder=7)
            ax.text(a[0] + 0.05, a[1] - 0.12, "$\\mathbf{a}$", fontsize=19, ha="left", va="top", color=C["red"])
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            despine(ax)
            if step == 2:
                L = 0.5
                for ang in np.radians([75, 115, 160, 250, 345]):
                    d = np.array([np.cos(ang), np.sin(ang)])
                    ax.annotate("", xy=a + L * d, xytext=a,
                                arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.6, mutation_scale=14))
                ax.plot(*np.c_[a - 0.6 * t, a + 0.6 * t], ls="--", color="k", lw=1.6)
                ax.annotate("", xy=a + L * n, xytext=a,
                            arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=3.2, mutation_scale=20))
                ax.annotate("", xy=a - L * n, xytext=a,
                            arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=3.2, mutation_scale=20))
                ax.text(*(a + (L + 0.06) * n), "$\\nabla f(\\mathbf{a})$", color=C["green"], fontsize=16,
                        va="bottom")
                ax.text(*(a - (L + 0.05) * n + np.array([-0.02, -0.06])), "$-\\nabla f(\\mathbf{a})$",
                        color=C["red"], fontsize=16, ha="right", va="top")
                ax.text(*(a + 0.62 * t + np.array([-0.05, 0.02])), "tangent to\ncontour", fontsize=12, ha="right")
            save(fig, f"gradient_direction_step{step}")


def fig_minus_sign():
    """One gradient-descent step on f(x) = (x - 1)^2, for positive and negative slope."""
    f = lambda x: (x - 1) ** 2  # noqa: E731
    alpha = 0.3
    xs = np.linspace(-0.05, 2.05, 200)
    for name, x0 in [("positive", 1.75), ("negative", 0.25)]:
        slope = 2 * (x0 - 1)
        x1 = x0 - alpha * slope
        with sketchy():
            fig, ax = plt.subplots(figsize=(6.0, 4.6))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.96)
            ax.plot(xs, f(xs), color=C["grey"], lw=2.6)
            tl = np.array([x0 - 0.32, x0 + 0.32])
            ax.plot(tl, f(x0) + slope * (tl - x0), color=C["orange"], lw=2.6)
            ax.scatter([x0], [f(x0)], s=110, color=C["red"], zorder=6)
            ax.scatter([x1], [f(x1)], s=110, color=C["blue"], zorder=6)
            ax.plot([x0, x0], [0, f(x0)], ls=":", color=C["red"], lw=1.6)
            ax.plot([x1, x1], [0, f(x1)], ls=":", color=C["blue"], lw=1.6)
            ax.annotate("", xy=(x1, 0.04), xytext=(x0, 0.04),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=2.6, mutation_scale=20))
            ax.text(x0, -0.1, "$x^{[0]}$", color=C["red"], ha="center", va="top", fontsize=17)
            ax.text(x1, -0.1, "$x^{[1]}$", color=C["blue"], ha="center", va="top", fontsize=17)
            sign = ">" if slope > 0 else "<"
            ax.text(1.0, 1.02, f"$f'(x^{{[0]}}) = {slope:g} {sign} 0$", ha="center", fontsize=17, color=C["orange"])
            ax.set_xlim(-0.05, 2.05)
            ax.set_ylim(-0.25, 1.15)
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x) = (x - 1)^2$")
            despine(ax)
            save(fig, f"gd_step_{name}_slope")


def fig_gd_in_action():
    """Toy data: gradient-descent path on the contours of J(w, b), and the corresponding lines."""
    alpha, n_iters = 0.3, 150
    w, b = -1.5, 6.0
    path = [(w, b)]
    m = len(TOY_X)
    for _ in range(n_iters):
        err = w * TOY_X + b - TOY_Y
        w, b = w - alpha * np.sum(err * TOY_X) / m, b - alpha * np.sum(err) / m
        path.append((w, b))
    path = np.array(path)
    f = lambda w, b: 7 / 3 * (w - 1) ** 2 + 2 * (w - 1) * b + b ** 2 / 2  # noqa: E731
    W, B = np.meshgrid(np.linspace(-2.2, 3.2, 200), np.linspace(-3.5, 7.5, 200))
    shown = [0, 1, 3, 10, 40, n_iters]
    cmap = plt.get_cmap("plasma")
    cols = [cmap(0.85 * k / (len(shown) - 1)) for k in range(len(shown))]
    with sketchy():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.92, wspace=0.25)
        cs = a1.contour(W, B, f(W, B), levels=[0.25, 1, 3, 6, 10, 16, 24], cmap="viridis", linewidths=1.8)
        a1.clabel(cs, fontsize=10, fmt="%g")
        a1.plot(path[:, 0], path[:, 1], color="k", lw=1.6, marker="o", ms=3.5)
        for k, c in zip(shown, cols):
            a1.scatter(*path[k], s=95, color=c, zorder=7, edgecolor="k", linewidths=0.8)
        a1.text(path[0, 0] + 0.15, path[0, 1] + 0.2, "start", fontsize=14)
        a1.scatter([1], [0], marker="*", s=260, color=C["red"], zorder=8)
        a1.set_xlabel("$w$")
        a1.set_ylabel("$b$")
        a1.set_title(f"path on contours of $J(w,b)$, $\\alpha = {alpha}$", fontsize=15)
        despine(a1)
        xs = np.array([0, 3.5])
        for k, c in zip(shown, cols):
            wk, bk = path[k]
            a2.plot(xs, wk * xs + bk, color=c, lw=2.6, label=f"$t = {k}$")
        data_points(a2, TOY_X, TOY_Y, s=120)
        a2.set_xlim(0, 3.5)
        a2.set_ylim(-1, 9.6)
        a2.set_xlabel("$x$")
        a2.set_ylabel("$y$")
        a2.set_title("model $\\hat{y} = w^{[t]}x + b^{[t]}$", fontsize=15)
        a2.legend(loc="upper right", fontsize=12, ncol=3, frameon=False, columnspacing=1.0)
        despine(a2)
        save(fig, "gd_in_action")


def _iterates(x0, alpha, n):
    xs = [x0]
    for _ in range(n):
        xs.append(xs[-1] - alpha * 2 * (xs[-1] - 1))
    return np.array(xs)


def _lr_panel(ax, x0, alpha, n=15, xlim=(-0.05, 2.05)):
    f = lambda x: (x - 1) ** 2  # noqa: E731
    g = np.linspace(*xlim, 300)
    ax.plot(g, f(g), color=C["light"], lw=3, zorder=1)
    it = _iterates(x0, alpha, n)
    ax.plot(it, f(it), color=C["blue"], lw=1.4, zorder=3)
    ax.scatter(it, f(it), s=34, color=C["red"], zorder=4)
    ax.scatter([it[0]], [f(it[0])], s=90, color=C["red"], zorder=5, edgecolor="k")
    ax.set_title(f"$\\alpha = {alpha:g}$")
    ax.text(0.5, 0.97, f"$x^{{[{n}]}} = {it[-1]:.4f}$", transform=ax.transAxes, ha="center", va="top",
            fontsize=16, color=C["navy"])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("$x$")
    despine(ax)


def fig_learning_rates():
    for name, x0, alphas in [("lr_small", 0.1, [0.025, 0.05, 0.1]), ("lr_large", 0.1, [0.5, 0.9, 1.0])]:
        with sketchy():
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
            fig.subplots_adjust(left=0.05, right=0.99, bottom=0.17, top=0.88, wspace=0.18)
            for ax, a in zip(axs, alphas):
                _lr_panel(ax, x0, a)
            save(fig, name)
    # divergence: path on the curve + distance to the minimum
    with sketchy():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.17, top=0.88, wspace=0.25)
        _lr_panel(a1, 0.95, 1.1)
        it = _iterates(0.95, 1.1, 15)
        a2.plot(np.arange(16), np.abs(it - 1), color=C["red"], marker="o", ms=5)
        a2.set_ylim(0, 0.85)
        a2.set_xlabel("iteration $t$")
        a2.set_title("distance to minimum $|x^{[t]} - 1|$")
        despine(a2)
        save(fig, "lr_diverge")
    # regimes: x[t] - 1 = (1 - 2 alpha)^t (x[0] - 1), one small panel per regime
    regimes = [(0.1, "monotone"), (0.5, "one step"), (0.9, "oscillates,\nconverges"), (1.0, "bounces\nforever"),
               (1.1, "diverges")]
    with sketchy():
        fig, axs = plt.subplots(1, 5, figsize=(12, 3.2), sharey=True)
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.86, wspace=0.12)
        t = np.arange(0, 11)
        for ax, (a, lab) in zip(axs, regimes):
            x_t = 1 + (1 - 2 * a) ** t * (0.1 - 1)
            ax.axhline(1, color="k", lw=1.2, ls=":")
            ax.plot(t, x_t, marker="o", ms=4, lw=2, color=C["blue"] if a < 1 else C["red"], clip_on=True)
            ax.set_title(f"$\\alpha = {a:g}$")
            ax.text(10, -0.75, lab, ha="right", va="bottom", fontsize=13)
            ax.set_ylim(-1.0, 3.0)
            ax.set_xticks([0, 5, 10])
            ax.set_xlabel("$t$")
            despine(ax)
        axs[0].set_ylabel("$x^{[t]}$")
        save(fig, "lr_regimes")


def fig_train_test():
    x, y = load_bmi()
    rng = np.random.default_rng(0)
    idx = rng.permutation(x.size)
    n_test = int(0.2 * x.size)
    te, tr = idx[:n_test], idx[n_test:]
    w, b = fit_line(x[tr], y[tr])
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.96)
        ax.scatter(x[tr], y[tr], marker="x", s=24, color=C["blue"], linewidths=1.3, alpha=0.7,
                   label=f"training set ({tr.size})")
        ax.scatter(x[te], y[te], marker="o", s=30, facecolor="none", edgecolor=C["orange"], linewidths=1.6,
                   label=f"test set ({te.size})")
        xs = np.array([17, 43])
        ax.plot(xs, w * xs + b, color="k", lw=2.8, label="fitted on training set")
        ax.set_xlabel("BMI")
        ax.set_ylabel("disease progression")
        ax.set_xlim(16, 44)
        ax.set_ylim(0, 450)
        ax.set_yticks([0, 100, 200, 300])
        ax.legend(loc="upper left", fontsize=13, frameon=False)
        despine(ax)
        save(fig, "train_test")


def fig_candidate_lines():
    """Three candidate lines through the BMI data: which w and b are good?"""
    x, y = load_bmi()
    w_ls, b_ls = fit_line(x, y)
    xs = np.linspace(17, 43, 100)
    lines = [(4.0, 60.0, C["orange"], "--"), (16.0, -300.0, C["purple"], "-."), (w_ls, b_ls, C["blue"], "-")]
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.96)
        ax.scatter(x, y, marker="x", s=28, color=C["red"], linewidths=1.4, alpha=0.75, zorder=3)
        for w, b, col, ls in lines:
            ax.plot(xs, w * xs + b, color=col, ls=ls, lw=3, zorder=4, label=f"$w = {w:.1f}$, $b = {b:.0f}$")
        ax.set_xlim(17, 43)
        ax.set_ylim(0, 380)
        ax.set_xlabel("BMI $x$")
        ax.set_ylabel("disease progression $y$")
        ax.legend(loc="upper left", fontsize=13, frameon=False)
        despine(ax)
        save(fig, "candidate_lines")
    print(f"    least squares: w = {w_ls:.2f}, b = {b_ls:.1f}")


# =========================================================================== part 2b
# --------------------------------------------------------------------------- data
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def load_diabetes_csv():
    """Return X (442 x 10, raw units) and y (disease progression)."""
    data = np.loadtxt(DATA_ALL, delimiter=",", skiprows=1)
    return data[:, :-1], data[:, -1]


def zscore(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def gradient_descent(X, y, alpha, n_iters, w0=None):
    """Batch gradient descent for f(x) = X @ w + b. Returns the path of w, the b values and J."""
    m, n = X.shape
    w = np.zeros(n) if w0 is None else np.array(w0, float)
    b = 0.0
    ws, bs, Js = [w.copy()], [b], []
    for _ in range(n_iters):
        err = X @ w + b - y
        Js.append((err ** 2).mean() / 2)
        w = w - alpha * X.T @ err / m
        b = b - alpha * err.mean()
        ws.append(w.copy())
        bs.append(b)
    err = X @ w + b - y
    Js.append((err ** 2).mean() / 2)
    return np.array(ws), np.array(bs), np.array(Js)


def optimal_cost(X, y):
    Xt = np.column_stack([np.ones(len(y)), X])
    theta = np.linalg.lstsq(Xt, y, rcond=None)[0]
    return ((Xt @ theta - y) ** 2).mean() / 2


def box(ax, xc, yc, text, w, h, fc="#FCE4EC", ec="k", fs=15, lw=2, color="k", round_=0.15):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle=f"round,pad=0.04,rounding_size={round_}", fc=fc, ec=ec, lw=lw))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, p, q, color="k", lw=2.0, ms=18, style="-|>"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=lw, color=color, mutation_scale=ms))


# --------------------------------------------------------------------------- figures
def fig_plane():
    """Two features (BMI, blood pressure): the linear model is a plane."""
    X, y = load_diabetes_csv()
    x1, x2 = X[:, 2], X[:, 3]
    A = np.column_stack([x1, x2, np.ones_like(x1)])
    w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]
    g1, g2 = np.meshgrid(np.linspace(18, 42, 12), np.linspace(62, 133, 12))
    with sketchy():
        fig = plt.figure(figsize=(7.0, 5.6))
        ax = fig.add_axes([0.0, 0.02, 0.95, 0.96], projection="3d")
        ax.computed_zorder = False
        ax.plot_surface(g1, g2, w1 * g1 + w2 * g2 + b, color=C["blue"], alpha=0.35, edgecolor=C["blue"], lw=0.6,
                        zorder=1)
        ax.scatter(x1, x2, y, marker="x", s=14, color=C["red"], alpha=0.7, depthshade=False, zorder=2)
        ax.set_xlabel("BMI $x_1$", labelpad=8)
        ax.set_ylabel("blood pressure $x_2$", labelpad=8)
        ax.set_zlabel("progression $y$", labelpad=6)
        ax.view_init(18, -60)
        ax.locator_params(nbins=4)
        for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            p.set_alpha(0.0)
        fig.text(0.04, 0.93, f"$\\hat y = {w1:.2f}\\,x_1 + {w2:.2f}\\,x_2 {b:+.1f}$", fontsize=16, color=C["blue"])
        save(fig, "plane_fit")


def fig_vectorization():
    """Loop: one multiply-add per time step. np.dot: all products at once, then one sum."""
    with sketchy():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1, 1.45]))
        fig.subplots_adjust(0.01, 0.02, 0.99, 0.9, wspace=0.08)
        for ax in (a1, a2):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")
        a1.set_title("loop: one step after another", color=C["red"])
        rows = [("$t_0$", "f += w[0]*x[0]"), ("$t_1$", "f += w[1]*x[1]"), ("$t_2$", "f += w[2]*x[2]"),
                ("", "$\\vdots$"), ("$t_{15}$", "f += w[15]*x[15]")]
        for k, (t, txt) in enumerate(rows):
            yc = 8.9 - 1.95 * k
            if t:
                a1.text(1.0, yc, t, fontsize=17, ha="center", va="center")
                box(a1, 5.6, yc, txt, 6.6, 1.15, fc="#FFF8E1", ec=C["orange"], fs=14)
            else:
                a1.text(5.6, yc, txt, fontsize=18, ha="center", va="center")
        a2.set_title("np.dot(w, x): all products at once", color=C["green"])
        cols = [1.3, 3.5, 5.7, 8.4]
        labels = ["0", "1", None, "15"]
        for xc, lab in zip(cols, labels):
            if lab is None:
                for yy in (8.6, 6.0):
                    a2.text(xc, yy, "$\\cdots$", fontsize=20, ha="center", va="center")
                continue
            box(a2, xc, 8.6, f"w[{lab}]", 1.7, 0.9, fc="#E3F2FD", ec=C["blue"], fs=14)
            box(a2, xc, 6.0, f"x[{lab}]", 1.7, 0.9, fc="#E3F2FD", ec=C["blue"], fs=14)
            a2.add_patch(plt.Circle((xc, 7.3), 0.38, fc="white", ec="k", lw=1.8))
            a2.text(xc, 7.3, "$\\times$", fontsize=15, ha="center", va="center")
            arrow(a2, (xc, 6.45), (xc, 4.15), lw=1.6, ms=14)
            arrow(a2, (xc, 3.65), (5.0, 2.0), lw=1.6, ms=14)
        a2.text(-0.2, 7.3, "$t_0$", fontsize=18, ha="center", va="center")
        a2.add_patch(plt.Circle((5.0, 1.55), 0.45, fc="white", ec="k", lw=2))
        a2.text(5.0, 1.55, "+", fontsize=20, ha="center", va="center")
        a2.text(-0.2, 1.55, "$t_1$", fontsize=18, ha="center", va="center")
        a2.text(5.8, 1.55, "fast for large $n$", fontsize=15, color=C["green"], va="center")
        save(fig, "vectorization")


def _profiled(X, y):
    """Centre features and target: then the best b is 0 and J depends on w only."""
    return X - X.mean(axis=0), y - y.mean()


def fig_scaling_contours():
    """Cost contours in (w1, w2) for two raw vs z-scored features, with 25 gradient-descent steps each."""
    X, y = load_diabetes_csv()
    pair = X[:, [4, 2]]                        # s1 (cholesterol, large values), bmi (small values)
    n_steps = 25
    panels = [("raw features", pair), ("z-score scaled features", zscore(pair))]
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.9, wspace=0.25)
        for ax, (title, F) in zip(axs, panels):
            Fc, yc = _profiled(F, y)
            m = len(yc)
            H = Fc.T @ Fc / m
            w_star = np.linalg.solve(H, Fc.T @ yc / m)
            alpha = 1.0 / np.linalg.eigvalsh(H)[-1]
            ws, _, _ = gradient_descent(Fc, yc, alpha, n_steps)
            half = 1.15 * max(np.abs(w_star).max(), 1e-9)
            c = w_star / 2
            g1, g2 = np.meshgrid(np.linspace(c[0] - half, c[0] + half, 300), np.linspace(c[1] - half, c[1] + half, 300))
            W = np.stack([g1.ravel(), g2.ravel()])
            d = W - w_star[:, None]
            Jg = 0.5 * np.einsum("ik,ij,jk->k", d, H, d).reshape(g1.shape)
            levels = np.quantile(Jg, [0.01, 0.04, 0.1, 0.2, 0.35, 0.55])
            ax.contour(g1, g2, Jg, levels=levels, cmap="viridis", linewidths=1.8)
            ax.plot(ws[:, 0], ws[:, 1], color=C["red"], lw=1.6, marker="o", ms=3.5)
            ax.scatter(*ws[0], s=70, color=C["red"], zorder=6)
            ax.scatter(*w_star, marker="*", s=280, color=C["navy"], zorder=7)
            ax.set_aspect("equal")
            ax.set_xlim(c[0] - half, c[0] + half)
            ax.set_ylim(c[1] - half, c[1] + half)
            ax.set_xlabel("$w_1$ (cholesterol S1)")
            ax.set_ylabel("$w_2$ (BMI)")
            ax.set_title(f"{title}: {n_steps} steps", fontsize=16)
            ax.locator_params(nbins=4)
            despine(ax)
        save(fig, "scaling_contours")


def fig_scaling_learning_curves():
    X, y = load_diabetes_csv()
    J_opt = optimal_cost(X, y)
    n = 100_000
    lam_raw = np.linalg.eigvalsh(np.cov(np.column_stack([np.ones(len(y)), X]).T, bias=True)
                                 + np.outer(np.r_[1, X.mean(0)], np.r_[1, X.mean(0)]))[-1]
    _, _, J_raw = gradient_descent(X, y, 1.9 / lam_raw, n)
    _, _, J_sc = gradient_descent(zscore(X), y, 0.3, n)
    t = np.arange(1, n + 2)
    with sketchy():
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.95)
        ax.loglog(t, J_raw - J_opt, color=C["red"], lw=2.8, label="raw features")
        ax.loglog(t, np.maximum(J_sc - J_opt, 1e-9), color=C["blue"], lw=2.8, label="z-score scaled features")
        ax.set_ylim(1e-4, 3e4)
        ax.set_xlim(1, n)
        ax.set_xlabel("iteration (log scale)")
        ax.set_ylabel("$J - J_{\\min}$ (log scale)")
        ax.legend(loc="lower left", frameon=False, fontsize=14)
        despine(ax)
        save(fig, "scaling_learning_curves")
        return J_raw, J_sc, J_opt


def _scatter_panel(ax, F, title, box_=None, lims=None, mean_mark=False, xlabel="", ylabel=""):
    ax.scatter(F[:, 0], F[:, 1], marker="x", s=18, color=C["red"], linewidths=1.3, alpha=0.75, zorder=4)
    if box_ is not None:
        (x0, x1), (y0, y1) = box_
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], ls="--", color=C["blue"], lw=2.2, zorder=3)
    if mean_mark:
        ax.scatter([0], [0], marker="*", s=260, color=C["navy"], zorder=6)
    ax.set_aspect("equal")
    if lims is not None:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=C["light"], lw=1, zorder=1)
    ax.axvline(0, color=C["light"], lw=1, zorder=1)
    despine(ax)


def fig_scaling_scatter():
    """Raw S1 vs BMI (equal aspect), and the three normalisations."""
    X, _ = load_diabetes_csv()
    F = X[:, [4, 2]]
    mn, mx, mu, sd = F.min(0), F.max(0), F.mean(0), F.std(0)
    minmax = (F - mn) / (mx - mn)
    meann = (F - mu) / (mx - mn)
    zs = (F - mu) / sd
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.2, top=0.88)
        _scatter_panel(ax, F, "raw data, same scale on both axes", lims=((80, 320), (0, 60)),
                       xlabel="$x_1$: cholesterol S1", ylabel="$x_2$: BMI")
        ax.locator_params(nbins=6)
        save(fig, "scaling_raw")
        lims = ((-2.6, 2.6), (-2.6, 2.6))
        specs = [("scaling_minmax", minmax, "min-max", ((0, 1), (0, 1)), False),
                 ("scaling_mean", meann, "mean normalisation", ((-1, 1), (-1, 1)), True),
                 ("scaling_zscore", zs, "z-score", None, True)]
        for name, G, title, bx, mm in specs:
            fig, ax = plt.subplots(figsize=(4.6, 4.6))
            fig.subplots_adjust(left=0.17, right=0.97, bottom=0.15, top=0.9)
            _scatter_panel(ax, G, title, box_=bx, lims=lims, mean_mark=mm,
                           xlabel="$x_{1,\\mathrm{scaled}}$", ylabel="$x_{2,\\mathrm{scaled}}$")
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            save(fig, name)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.14, top=0.88, wspace=0.22)
        for ax, (_, G, title, bx, mm) in zip(axs, specs):
            _scatter_panel(ax, G, title, box_=bx, lims=lims, mean_mark=mm)
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
        save(fig, "scaling_comparison")


def fig_learning_curve():
    """A healthy learning curve (scaled diabetes data, alpha = 0.01) with three marked iterations."""
    X, y = load_diabetes_csv()
    _, _, J = gradient_descent(zscore(X), y, 0.01, 500)
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.95)
        ax.plot(np.arange(len(J)), J, color=C["blue"], lw=3)
        for k, dy, txt in [(100, 5200, "after 100 iterations"), (200, 4200, "after 200"),
                           (400, 3600, "after 400:\nhardly changes")]:
            ax.scatter([k], [J[k]], s=80, color=C["red"], zorder=6)
            ax.annotate(f"$J = {J[k]:.0f}$ {txt}", xy=(k, J[k]), xytext=(k - 60, J[k] + dy), fontsize=13,
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="k"))
        ax.set_xlabel("iteration")
        ax.set_ylabel("$J(\\mathbf{w}, b)$")
        ax.set_ylim(0, 16000)
        ax.set_xlim(0, 500)
        despine(ax)
        save(fig, "learning_curve")
        return J


def fig_bad_learning_curves():
    """Real runs on the scaled diabetes data: a sign bug (w + alpha*grad), and alpha slightly too large."""
    X, y = load_diabetes_csv()
    Z = zscore(X)
    _, _, J_bug = gradient_descent(Z, y, -0.01, 300)       # same as w = w + 0.01 * dJ/dw: the sign bug
    _, _, J_edge = gradient_descent(Z, y, 0.5, 150)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.88, wspace=0.28)
        axs[0].semilogy(J_bug, color=C["red"], lw=3)
        axs[0].set_title("$J$ increases from the start", fontsize=15)
        axs[0].set_ylabel("$J(\\mathbf{w}, b)$, log scale")
        axs[1].plot(J_edge, color=C["red"], lw=3)
        axs[1].set_ylim(0, 16000)
        axs[1].set_title("$J$ decreases, then explodes", fontsize=15)
        axs[1].set_ylabel("$J(\\mathbf{w}, b)$")
        for ax in axs:
            ax.set_xlabel("iteration")
            despine(ax)
        save(fig, "bad_learning_curves")


def fig_lr_sweep():
    X, y = load_diabetes_csv()
    Z = zscore(X)
    J_opt = optimal_cost(X, y)
    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    cmap = plt.get_cmap("viridis")
    with sketchy():
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        fig.subplots_adjust(left=0.13, right=0.8, bottom=0.15, top=0.95)
        for k, a in enumerate(alphas):
            _, _, J = gradient_descent(Z, y, a, 300)
            col = cmap(0.9 * k / (len(alphas) - 1))
            ax.plot(J, color=col, lw=2.6, label=f"$\\alpha = {a:g}$")
        _, _, J = gradient_descent(Z, y, 0.5, 300)
        ax.plot(J, color=C["red"], lw=2.6, ls="--", label="$\\alpha = 0.5$")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=13)
        ax.axhline(J_opt, color="k", ls=":", lw=1.4)
        ax.set_ylim(0, 16500)
        ax.set_xlim(0, 300)
        ax.set_xlabel("iteration")
        ax.set_ylabel("$J(\\mathbf{w}, b)$")
        despine(ax)
        save(fig, "lr_sweep")


def fig_normal_equation_example():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([300.0, 450.0, 500.0])
    b, w = 650 / 3, 100.0
    with sketchy():
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.15, top=0.95)
        xs = np.array([0.5, 3.5])
        ax.plot(xs, w * xs + b, color=C["blue"], lw=3)
        data_points(ax, x, y, s=130)
        ax.text(0.6, 545, "$\\hat y = 100\\,x + 216.67$", color=C["blue"], fontsize=16)
        ax.set_xlim(0.4, 3.6)
        ax.set_ylim(240, 590)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        despine(ax)
        save(fig, "normal_equation_example")


def fig_feature_engineering_area():
    with sketchy():
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(-0.6, 7.2)
        ax.set_ylim(-1.6, 5.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.6, 0.6), 5.0, 3.8, fc="#FCE4EC", ec="k", lw=2.4))
        ax.add_patch(plt.Rectangle((1.8, 1.6), 2.2, 1.6, fc="white", ec=C["grey"], lw=1.6))
        ax.plot([1.6, 2.9, 4.2], [3.2, 4.0, 3.2], color=C["grey"], lw=1.6)
        ax.text(3.1, 2.15, "house", fontsize=13, color=C["grey"], ha="center")
        ax.plot([-0.4, 6.8], [-0.2, -0.2], color=C["grey"], lw=6, alpha=0.5)
        ax.text(6.7, -0.75, "street", fontsize=13, color=C["grey"], ha="right")
        arrow(ax, (0.6, 0.15), (5.6, 0.15), style="<|-|>", lw=1.8)
        ax.text(3.1, -1.05, "frontage $x_1$", fontsize=16, ha="center", color=C["blue"])
        arrow(ax, (6.0, 0.6), (6.0, 4.4), style="<|-|>", lw=1.8)
        ax.text(6.2, 2.5, "depth\n$x_2$", fontsize=16, va="center", color=C["blue"])
        ax.text(3.1, 4.85, "area $x_3 = x_1 x_2$", fontsize=17, ha="center", color=C["red"])
        save(fig, "feature_engineering_area")


def fig_feature_engineering_pipeline():
    with sketchy():
        fig, ax = plt.subplots(figsize=(12, 3.4))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3.4)
        ax.axis("off")
        for k, col in enumerate([C["blue"], C["green"], C["red"]]):
            yc = 2.6 - 0.8 * k
            ax.add_patch(plt.Circle((0.5, yc), 0.22, fc=col, ec="k", lw=1.5))
            ax.text(0.85, yc, f"source {k + 1}" if k < 2 else "source n", fontsize=12, va="center")
            arrow(ax, (1.9, yc), (2.55, 1.8), lw=1.5, ms=14)
        steps = [(3.3, "raw\ndata", "#FCE4EC"), (5.45, "clean,\ntransform,\ncombine", "#FFF8E1"),
                 (7.6, "features", "#E3F2FD"), (9.4, "model", "#E8F5E9"), (11.1, "predictions", "#F3E5F5")]
        for k, (xc, txt, fc) in enumerate(steps):
            w = 1.45 if k != 1 else 1.75
            box(ax, xc, 1.8, txt, w, 1.25, fc=fc, ec="k" if k != 1 else C["red"], fs=14, lw=2 if k != 1 else 3)
            if k:
                prev = steps[k - 1][0] + (0.73 if k != 2 else 0.88)
                arrow(ax, (prev, 1.8), (xc - (0.73 if k != 1 else 0.88), 1.8), lw=1.8)
        ax.text(5.45, 0.55, "feature engineering", fontsize=15, ha="center", color=C["red"])
        save(fig, "feature_engineering_pipeline")


def _poly_data():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.2, 10, 60))
    y = 8 - 6 * np.exp(-0.4 * x) + rng.normal(0, 0.4, x.size)
    return x, y


def fig_polynomial():
    x, y = _poly_data()
    xs = np.linspace(0, 10.2, 300)
    labels = [None, "$\\hat y = w_1 x + b$", "$\\hat y = w_1 x + w_2 x^2 + b$",
              "$\\hat y = w_1 x + w_2 x^2 + w_3 x^3 + b$"]
    for step, deg in enumerate([0, 1, 2, 3], start=1):
        with sketchy():
            fig, ax = plt.subplots(figsize=(8, 4.6))
            fig.subplots_adjust(left=0.06, right=0.97, bottom=0.1, top=0.97)
            arrow_axes(ax, xlim=(0, 10.6), ylim=(0, 10))
            data_points(ax, x, y, s=70)
            if deg:
                coef = np.polyfit(x, y, deg)
                ax.plot(xs, np.polyval(coef, xs), color=C["blue"], lw=3.2)
                ax.text(0.4, 9.2, labels[deg], fontsize=18, color=C["blue"])
            save(fig, f"poly_step{step}")


def fig_r2():
    """SS_tot: squares to the mean; SS_res: squares to the fitted line."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.6, 2.2, 3.9, 3.6, 5.2])
    w, b = np.polyfit(x, y, 1)
    yb = y.mean()
    ss_tot = ((y - yb) ** 2).sum()
    ss_res = ((y - (w * x + b)) ** 2).sum()
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.88, wspace=0.12)
        for ax, ref, col, title in [
            (axs[0], np.full_like(x, yb), C["red"], f"$SS_{{\\mathrm{{tot}}}} = {ss_tot:.2f}$: squares to the mean $\\bar y$"),
            (axs[1], w * x + b, C["blue"], f"$SS_{{\\mathrm{{res}}}} = {ss_res:.2f}$: squares to the model"),
        ]:
            for xi, yi, ri in zip(x, y, ref):
                s = yi - ri
                ax.add_patch(plt.Rectangle((xi, min(yi, ri)), abs(s), abs(s), fc=col, alpha=0.25, ec=col, lw=1.5))
                ax.plot([xi, xi], [yi, ri], color=col, lw=1.8)
            if ref[0] == ref[-1]:
                ax.axhline(yb, color=C["red"], lw=2.4)
                ax.text(0.45, yb + 0.12, "$\\bar y$", fontsize=17, color=C["red"])
            else:
                xx = np.array([0.4, 6.0])
                ax.plot(xx, w * xx + b, color=C["blue"], lw=2.4)
            data_points(ax, x, y, s=110)
            ax.set_aspect("equal")
            ax.set_xlim(0.3, 6.3)
            ax.set_ylim(0.3, 6.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=15)
            despine(ax)
        fig.text(0.5, 0.01, f"$R^2 = 1 - SS_{{\\mathrm{{res}}}}/SS_{{\\mathrm{{tot}}}} = {1 - ss_res / ss_tot:.2f}$",
                 ha="center", fontsize=16)
        save(fig, "r2_illustration")


def fig_sklearn_1d():
    """Output of the scikit-learn 1D example on the slides (same code and seed)."""
    from_numpy = np.random.default_rng(42)
    w_true, b_true = 3.0, 1.0
    x_train = from_numpy.normal(loc=10.0, scale=3.0, size=100)
    y_train = w_true * x_train + b_true + from_numpy.normal(0.0, 1.0, size=100)
    w, b = np.polyfit(x_train, y_train, 1)
    with sketchy():
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        xx = np.linspace(0, 20, 101)
        ax.plot(xx, w * xx + b, "k-", lw=2.6)
        ax.scatter(x_train, y_train, marker="x", s=30, color=C["red"], linewidths=1.6)
        ax.text(0.5, 58, f"$w = {w:.3f}$,  $b = {b:.3f}$", fontsize=15)
        ax.set_xlabel("x_train")
        ax.set_ylabel("y_train")
        despine(ax)
        save(fig, "sklearn_1d_example")
        return w, b


FIGURES = {   # name: (output folder, function)
    "bmi": (OUT_A, fig_bmi),
    "pipeline": (OUT_A, fig_pipeline),
    "slope_intercept": (OUT_A, fig_slope_intercept),
    "residuals_two_points": (OUT_A, fig_residuals_two_points),
    "residuals_many": (OUT_A, fig_residuals_many),
    "abs_vs_squared": (OUT_A, fig_abs_vs_squared),
    "toy_cost": (OUT_A, fig_toy_cost_build),
    "bowl": (OUT_A, fig_bowl),
    "toy_cost_surface": (OUT_A, fig_toy_cost_surface),
    "stationary_1d": (OUT_A, fig_stationary_1d),
    "stationary_2d": (OUT_A, fig_stationary_2d),
    "gd_landscape": (OUT_A, fig_gd_landscape),
    "gradient_direction": (OUT_A, fig_gradient_direction),
    "minus_sign": (OUT_A, fig_minus_sign),
    "gd_in_action": (OUT_A, fig_gd_in_action),
    "lr": (OUT_A, fig_learning_rates),
    "train_test": (OUT_A, fig_train_test),
    "candidate_lines": (OUT_A, fig_candidate_lines),
    "plane": (OUT_B, fig_plane),
    "vectorization": (OUT_B, fig_vectorization),
    "scaling_contours": (OUT_B, fig_scaling_contours),
    "scaling_learning_curves": (OUT_B, fig_scaling_learning_curves),
    "scaling_scatter": (OUT_B, fig_scaling_scatter),
    "learning_curve": (OUT_B, fig_learning_curve),
    "bad_learning_curves": (OUT_B, fig_bad_learning_curves),
    "lr_sweep": (OUT_B, fig_lr_sweep),
    "normal_equation": (OUT_B, fig_normal_equation_example),
    "feature_engineering_area": (OUT_B, fig_feature_engineering_area),
    "feature_engineering_pipeline": (OUT_B, fig_feature_engineering_pipeline),
    "polynomial": (OUT_B, fig_polynomial),
    "r2": (OUT_B, fig_r2),
    "sklearn_1d": (OUT_B, fig_sklearn_1d),
}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}")
    for key, (folder, fn) in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            OUT = folder                    # save() writes to the folder of this figure
            print(f"{key}  ->  {folder.relative_to(ROOT)}")
            fn()
