"""
Sketch-style figures for Lecture 3: Logistic regression.
    Part 3a, binary classification     -> figures/logistic-regression-binary-sketch/
    Part 3b, multiclass classification -> figures/logistic-regression-multiclass-sketch/

Run from anywhere:
    python scripts/make_figures_lecture03.py                    # all figures
    python scripts/make_figures_lecture03.py sigmoid softmax    # only figures whose name contains these words

Requirements: numpy, matplotlib >= 3.7, scikit-learn. All data come with scikit-learn or are synthetic:
    breast-cancer data (Wolberg et al., UCI) via sklearn.datasets.load_breast_cancer, with y = 1 for
    malignant and y = 0 for benign (part 3a); iris data via sklearn.datasets.load_iris (part 3b);
    synthetic data from make_blobs / make_moons with fixed random seeds.

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
OUT_A = ROOT / "figures" / "logistic-regression-binary-sketch"       # part 3a
OUT_B = ROOT / "figures" / "logistic-regression-multiclass-sketch"   # part 3b
OUT = OUT_A                                                          # set per figure in __main__
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


# =========================================================================== part 3a
# --------------------------------------------------------------------------- data and helpers
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def box(ax, xc, yc, text, w, h, fc="#FCE4EC", ec="k", fs=15, lw=2, color="k", round_=0.15):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle=f"round,pad=0.04,rounding_size={round_}", fc=fc, ec=ec, lw=lw))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, p, q, color="k", lw=2.0, ms=18, style="-|>"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=lw, color=color, mutation_scale=ms))


def neg_points(ax, x, y, s=110, **kw):
    """Negative class (y = 0): open circles."""
    ax.scatter(x, y, marker="o", s=s, facecolor="white", edgecolor=C["blue"], linewidths=2.2, zorder=5, **kw)


def pos_points(ax, x, y, s=110, **kw):
    """Positive class (y = 1): crosses."""
    ax.scatter(x, y, marker="x", s=s, color=C["red"], linewidths=2.6, zorder=5, **kw)


def class_axes(ax, xlabel="tumour size $x$ (cm)", xlim=(0, 11)):
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.3, 1.45)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0 (no)", "0.5", "1 (yes)"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("malignant?")
    despine(ax)


def load_breast_cancer_xy():
    """Breast-cancer data with y = 1 for malignant, y = 0 for benign."""
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer()
    return d.data, 1 - d.target, list(d.feature_names)


# Toy 1D data: tumour sizes in cm (perfectly separable)
X_BENIGN = np.array([0.6, 1.0, 1.4, 1.8, 2.2])
X_MALIGN = np.array([3.0, 3.4, 3.8, 4.2])
X_LARGE = np.array([9.0, 10.5])


# --------------------------------------------------------------------------- figures
def fig_linreg_fail():
    """Why not linear regression? The fitted line and its 0.5 boundary move when large tumours are added."""
    x1 = np.r_[X_BENIGN, X_MALIGN]
    y1 = np.r_[np.zeros(X_BENIGN.size), np.ones(X_MALIGN.size)]
    x2 = np.r_[x1, X_LARGE]
    y2 = np.r_[y1, np.ones(X_LARGE.size)]
    w1, b1 = np.polyfit(x1, y1, 1)
    w2, b2 = np.polyfit(x2, y2, 1)
    d1, d2 = (0.5 - b1) / w1, (0.5 - b2) / w2
    xs = np.array([0, 11])
    for step in (1, 2, 3, 4):
        with sketchy():
            fig, ax = plt.subplots(figsize=(9, 4.6))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.96)
            class_axes(ax)
            neg_points(ax, X_BENIGN, np.zeros(X_BENIGN.size))
            pos_points(ax, X_MALIGN, np.ones(X_MALIGN.size))
            if step >= 3:
                pos_points(ax, X_LARGE, np.ones(X_LARGE.size))
            if step in (2, 3):
                ax.plot(xs, w1 * xs + b1, color=C["blue"] if step == 2 else C["light"], lw=2.6)
                ax.axhline(0.5, color=C["purple"], ls="--", lw=1.6)
                ax.axvline(d1, color=C["green"] if step == 2 else C["light"], lw=2.6)
                if step == 2:
                    ax.text(d1 + 0.15, 1.3, "decision boundary", color=C["green"], fontsize=15)
                    ax.text(d1 - 0.25, 0.75, "$\\hat y = 0$", ha="right", fontsize=16)
                    ax.text(d1 + 0.25, 0.22, "$\\hat y = 1$", ha="left", fontsize=16)
                    ax.text(0.3, 1.25, "$f_{w,b}(x) = wx + b$", color=C["blue"], fontsize=15)
            if step == 3:
                ax.text(9.75, 1.22, "two large tumours", ha="center", fontsize=14, color=C["red"])
            if step == 4:
                ax.plot(xs, w1 * xs + b1, color=C["light"], lw=2.2)
                ax.axvline(d1, color=C["light"], lw=2.2)
                ax.plot(xs, w2 * xs + b2, color=C["blue"], lw=2.6)
                ax.axhline(0.5, color=C["purple"], ls="--", lw=1.6)
                ax.axvline(d2, color=C["green"], lw=2.6)
                ax.text(d2 + 0.15, 1.3, "new boundary", color=C["green"], fontsize=15)
                wrong = X_MALIGN[X_MALIGN < d2]
                ax.scatter(wrong, np.ones_like(wrong), s=700, facecolor="none", edgecolor=C["purple"], linewidths=2.2,
                           zorder=6)
                ax.annotate("now predicted $\\hat y = 0$!", xy=(wrong.mean(), 1.08), xytext=(0.3, 1.25),
                            fontsize=15, color=C["purple"],
                            arrowprops=dict(arrowstyle="-|>", color=C["purple"], lw=1.8))
            save(fig, f"linreg_fail_step{step}")
    print(f"    boundaries: before {d1:.2f}, after {d2:.2f}")


def fig_sigmoid():
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.88, wspace=0.25)
        for ax, lim, title in [(axs[0], 6, "$g(z) = \\frac{1}{1 + e^{-z}}$"), (axs[1], 100, "zoomed out: almost a step")]:
            z = np.linspace(-lim, lim, 600)
            ax.axhline(1, color="k", ls=":", lw=1.4)
            ax.axhline(0, color="k", ls=":", lw=1.4)
            ax.plot(z, sigmoid(z), color=C["red"], lw=3)
            ax.set_ylim(-0.08, 1.12)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("$z$")
            ax.set_title(title, fontsize=17)
            despine(ax)
        axs[0].scatter([0], [0.5], s=90, color=C["navy"], zorder=6)
        axs[0].annotate("$g(0) = 0.5$", xy=(0, 0.5), xytext=(1.0, 0.25), fontsize=15,
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k"))
        axs[0].set_ylabel("$g(z)$")
        save(fig, "sigmoid")


def fig_model_diagram():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11, 2.4))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 2.4)
        ax.axis("off")
        box(ax, 0.8, 1.2, "$\\mathbf{x}$", 1.0, 0.9, fc="#FFF8E1", ec=C["orange"], fs=20)
        box(ax, 3.4, 1.2, "$z = \\mathbf{w}\\cdot\\mathbf{x} + b$", 2.9, 0.95, fc="#E3F2FD", ec=C["blue"], fs=18)
        box(ax, 7.0, 1.2, "$g(z) = \\frac{1}{1 + e^{-z}}$", 2.9, 1.15, fc="#FCE4EC", ec=C["red"], fs=18)
        box(ax, 10.0, 1.2, "$f_{\\mathbf{w},b}(\\mathbf{x})$", 1.6, 0.95, fc="#E8F5E9", ec=C["green"], fs=18)
        arrow(ax, (1.32, 1.2), (1.92, 1.2))
        arrow(ax, (4.88, 1.2), (5.52, 1.2))
        ax.text(5.2, 1.42, "$z$", ha="center", fontsize=17)
        arrow(ax, (8.48, 1.2), (9.17, 1.2))
        ax.text(3.4, 0.28, "linear part, as in linear regression", ha="center", fontsize=13, color=C["blue"])
        ax.text(7.0, 0.28, "squashes $z$ into $(0, 1)$", ha="center", fontsize=13, color=C["red"])
        save(fig, "model_diagram")


def _toy_logistic_fit():
    from sklearn.linear_model import LogisticRegression

    x = np.r_[X_BENIGN, X_MALIGN, X_LARGE]
    y = np.r_[np.zeros(X_BENIGN.size), np.ones(X_MALIGN.size + X_LARGE.size)]
    model = LogisticRegression().fit(x.reshape(-1, 1), y)       # scikit-learn's default settings
    return x, y, model.coef_[0, 0], model.intercept_[0]


def fig_logistic_fit():
    """The same data as the linear-regression example, fitted with a sigmoid."""
    x, y, w, b = _toy_logistic_fit()
    xs = np.linspace(0, 11, 400)
    x07 = (np.log(0.7 / 0.3) - b) / w
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(9, 4.6))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.96)
            class_axes(ax)
            neg_points(ax, x[y == 0], np.zeros((y == 0).sum()))
            pos_points(ax, x[y == 1], np.ones((y == 1).sum()))
            ax.plot(xs, sigmoid(w * xs + b), color="k", lw=3)
            ax.text(5.3, 0.62, "$f_{w,b}(x) = g(wx + b)$", fontsize=16)
            if step == 2:
                ax.plot([0, x07], [0.7, 0.7], color=C["purple"], ls="--", lw=1.8)
                ax.plot([x07, x07], [-0.3, 0.7], color=C["purple"], ls="--", lw=1.8)
                ax.scatter([x07], [0.7], s=90, color=C["purple"], zorder=7)
                ax.text(x07 + 0.2, 0.33, f"$x = {x07:.1f}$ cm:\n70% chance malignant", fontsize=14,
                        color=C["purple"])
            save(fig, f"logistic_fit_step{step}")
    print(f"    toy logistic fit: w = {w:.3f}, b = {b:.3f}, boundary {-b / w:.2f}, f=0.7 at {x07:.2f}")


def fig_sigmoid_threshold():
    z = np.linspace(-6, 6, 400)
    with sketchy():
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        ax.axvspan(-6, 0, color=C["blue"], alpha=0.08)
        ax.axvspan(0, 6, color=C["red"], alpha=0.08)
        ax.axhline(0.5, color=C["purple"], ls="--", lw=1.8)
        ax.axvline(0, color="k", lw=1.2)
        ax.plot(z, sigmoid(z), color=C["red"], lw=3)
        ax.text(-5.6, 0.75, "$z < 0$\n$g(z) < 0.5$\n$\\hat y = 0$", fontsize=16, color=C["blue"])
        ax.text(3.0, 0.12, "$z \\geq 0$\n$g(z) \\geq 0.5$\n$\\hat y = 1$", fontsize=16, color=C["red"])
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("$z = \\mathbf{w}\\cdot\\mathbf{x} + b$")
        ax.set_ylabel("$g(z)$")
        despine(ax)
        save(fig, "sigmoid_threshold")


def fig_threshold_radius():
    """Real data: malignant vs mean radius, fitted sigmoid, boundaries for three thresholds."""
    from sklearn.linear_model import LogisticRegression

    X, y, _ = load_breast_cancer_xy()
    r = X[:, 0]
    model = LogisticRegression(C=1e6, max_iter=1000).fit(r.reshape(-1, 1), y)   # (practically) unregularised
    w, b = model.coef_[0, 0], model.intercept_[0]
    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.06, 0.06, r.size)
    xs = np.linspace(5, 30, 400)
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(9, 4.6))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.96)
            ax.scatter(r[y == 0], y[y == 0] + jitter[y == 0], marker="o", s=22, facecolor="none",
                       edgecolor=C["blue"], alpha=0.55, linewidths=1.2)
            ax.scatter(r[y == 1], y[y == 1] + jitter[y == 1], marker="x", s=22, color=C["red"], alpha=0.6,
                       linewidths=1.3)
            ax.plot(xs, sigmoid(w * xs + b), color="k", lw=3)
            taus = [0.5] if step == 1 else [0.3, 0.5, 0.7]
            for tau in taus:
                xb = (np.log(tau / (1 - tau)) - b) / w
                col = C["green"] if tau == 0.5 else C["purple"]
                ax.axvline(xb, color=col, lw=2.4 if tau == 0.5 else 2.0, ls="-" if tau == 0.5 else "--")
                ax.text(xb + (0.15 if tau >= 0.5 else -0.15), 1.3 if tau == 0.5 else (1.18 if tau > 0.5 else 1.06),
                        f"$\\tau = {tau:g}$: {xb:.1f}", color=col, fontsize=13,
                        ha="left" if tau >= 0.5 else "right")
            ax.set_xlim(5, 30)
            ax.set_ylim(-0.25, 1.42)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0 (benign)", "0.5", "1 (malignant)"])
            ax.set_xlabel("mean radius of the cell nuclei $x$")
            despine(ax)
            save(fig, f"threshold_radius_step{step}")
    print(f"    radius model: w = {w:.4f}, b = {b:.4f}, boundary {-b / w:.2f}")


LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


def _region_plot(ax, decision, xlim, ylim, alpha=0.1):
    g1, g2 = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = decision(g1, g2)
    ax.contourf(g1, g2, (Z >= 0).astype(float), levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]],
                alpha=alpha)
    ax.contour(g1, g2, Z, levels=[0], colors=[C["green"]], linewidths=3)


def fig_boundary_linear():
    rng = np.random.default_rng(4)
    P = rng.uniform(0.1, 3.0, size=(60, 2))
    s = P.sum(axis=1)
    P = P[np.abs(s - 3) > 0.35][:26]
    s = P.sum(axis=1)
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            fig.subplots_adjust(left=0.15, right=0.97, bottom=0.13, top=0.97)
            if step == 2:
                _region_plot(ax, lambda a, b: a + b - 3, (0, 3.2), (0, 3.2))
                ax.text(2.0, 2.75, "$\\hat y = 1$", fontsize=17, color=C["red"], bbox=LABEL_BOX, zorder=8)
                ax.text(0.15, 0.25, "$\\hat y = 0$", fontsize=17, color=C["blue"], bbox=LABEL_BOX, zorder=8)
                ax.text(0.5, 2.9, "$x_1 + x_2 = 3$", fontsize=16, color=C["green"], rotation=0, bbox=LABEL_BOX, zorder=8)
            neg_points(ax, P[s < 3, 0], P[s < 3, 1], s=80)
            pos_points(ax, P[s >= 3, 0], P[s >= 3, 1], s=80)
            ax.set_xlim(0, 3.2)
            ax.set_ylim(0, 3.2)
            ax.set_aspect("equal")
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2, 3])
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            despine(ax)
            save(fig, f"boundary_linear_step{step}")


def fig_boundary_ellipse():
    rng = np.random.default_rng(7)
    P = rng.uniform(-2.4, 2.4, size=(160, 2))
    q = P[:, 0] ** 2 + 2 * P[:, 1] ** 2
    P = P[np.abs(q - 3) > 0.6][:60]
    q = P[:, 0] ** 2 + 2 * P[:, 1] ** 2
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            fig.subplots_adjust(left=0.15, right=0.97, bottom=0.13, top=0.97)
            if step == 2:
                _region_plot(ax, lambda a, b: a ** 2 + 2 * b ** 2 - 3, (-2.5, 2.5), (-2.5, 2.5))
                ax.text(-1.25, -0.95, "$\\hat y = 0$", fontsize=17, color=C["blue"], bbox=LABEL_BOX, zorder=8)
                ax.text(1.0, 2.05, "$\\hat y = 1$", fontsize=17, color=C["red"], bbox=LABEL_BOX, zorder=8)
                ax.text(-2.4, -2.35, "$x_1^2 + 2x_2^2 = 3$", fontsize=15, color=C["green"], bbox=LABEL_BOX, zorder=8)
            neg_points(ax, P[q < 3, 0], P[q < 3, 1], s=70)
            pos_points(ax, P[q >= 3, 0], P[q >= 3, 1], s=70)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            despine(ax)
            save(fig, f"boundary_ellipse_step{step}")


def fig_boundary_wiggly():
    """make_moons: linear features vs degree-5 polynomial features (both fitted with scikit-learn)."""
    from sklearn.datasets import make_moons
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    X, y = make_moons(n_samples=160, noise=0.22, random_state=3)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.88, wspace=0.12)
        for ax, deg in zip(axs, [1, 5]):
            model = make_pipeline(PolynomialFeatures(deg, include_bias=False), StandardScaler(),
                                  LogisticRegression(C=100, max_iter=5000)).fit(X, y)
            _region_plot(ax, lambda a, b: model.decision_function(np.c_[a.ravel(), b.ravel()]).reshape(a.shape),
                         (-1.6, 2.6), (-1.2, 1.7), alpha=0.12)
            neg_points(ax, X[y == 0, 0], X[y == 0, 1], s=45)
            pos_points(ax, X[y == 1, 0], X[y == 1, 1], s=45)
            acc = model.score(X, y)
            ax.set_title(("features $x_1, x_2$" if deg == 1 else "all monomials up to degree 5")
                         + f": accuracy {acc:.0%}", fontsize=15)
            ax.set_xlim(-1.6, 2.6)
            ax.set_ylim(-1.2, 1.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            despine(ax)
        save(fig, "boundary_wiggly")


def _cost_toy():
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1], dtype=float)
    return x, y


def fig_cost_surfaces():
    """Same toy data, same sigmoid model: squared-error cost (non-convex) vs logistic cost (convex)."""
    x, y = _cost_toy()
    W, B = np.meshgrid(np.linspace(-4, 6, 140), np.linspace(-20, 10, 140))
    Fz = sigmoid(W[..., None] * x + B[..., None])
    J_sq = ((Fz - y) ** 2).mean(axis=-1) / 2
    eps = 1e-12
    J_log = -(y * np.log(Fz + eps) + (1 - y) * np.log(1 - Fz + eps)).mean(axis=-1)
    with sketchy():
        fig = plt.figure(figsize=(11.5, 4.6))
        for k, (J, title, col) in enumerate([(J_sq, "squared error: not convex", C["red"]),
                                             (J_log, "logistic loss: convex", C["green"])]):
            ax = fig.add_axes([0.0 + 0.5 * k, 0.0, 0.5, 0.92], projection="3d")
            ax.plot_surface(W, B, J, cmap="viridis", rstride=6, cstride=6, edgecolor="k", lw=0.25, alpha=0.9)
            ax.view_init(28, -125 if k == 0 else -60)
            ax.set_xlabel("$w$")
            ax.set_ylabel("$b$")
            ax.set_zlabel("$J(w, b)$")
            ax.locator_params(nbins=4)
            for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                p.set_alpha(0.0)
            fig.text(0.25 + 0.5 * k, 0.93, title, ha="center", fontsize=17, color=col)
        save(fig, "cost_surfaces")


def fig_loss_curves():
    f = np.linspace(0.004, 0.996, 400)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.25)
        for ax, loss, title, good, bad in [
            (axs[0], -np.log(f), "true label $y = 1$: loss $-\\log(f)$", 0.97, 0.03),
            (axs[1], -np.log(1 - f), "true label $y = 0$: loss $-\\log(1 - f)$", 0.03, 0.97),
        ]:
            ax.plot(f, loss, color=C["blue"], lw=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 5.5)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xlabel("prediction $f = f_{\\mathbf{w},b}(\\mathbf{x})$")
            ax.set_ylabel("loss $L$")
            ax.set_title(title, fontsize=16)
            ax.scatter([good], [0.06], s=110, color=C["green"], zorder=6)
            ax.text(good, 0.55, "right and confident:\nloss near 0", fontsize=13, color=C["green"],
                    ha="right" if good > 0.5 else "left")
            ax.annotate("wrong and confident:\nloss $\\to \\infty$", xy=(bad, 4.9), xytext=(0.35, 4.0), fontsize=13,
                        color=C["red"], ha="center", arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.6))
            despine(ax)
        save(fig, "loss_curves")


def _blobs():
    rng = np.random.default_rng(11)
    X0 = rng.normal([-1.0, -0.6], 0.8, size=(40, 2))
    X1 = rng.normal([1.0, 0.8], 0.8, size=(40, 2))
    return np.r_[X0, X1], np.r_[np.zeros(40), np.ones(40)]


def logistic_gd(X, y, alpha=0.5, n_iters=1000, record=()):
    """Vectorised batch gradient descent for logistic regression (the code on the slides)."""
    m, n = X.shape
    w, b = np.zeros(n), 0.0
    J_history, snapshots = [], {}
    for t in range(n_iters + 1):
        if t in record:
            snapshots[t] = (w.copy(), b)
        f = sigmoid(X @ w + b)
        J_history.append(-np.mean(y * np.log(f) + (1 - y) * np.log(1 - f)))
        if t == n_iters:
            break
        err = f - y
        w = w - alpha * X.T @ err / m
        b = b - alpha * err.mean()
    return w, b, np.array(J_history), snapshots


def fig_gd_training():
    """Gradient descent on two overlapping classes: learning curve and the trained model."""
    X, y = _blobs()
    w, b, J, _ = logistic_gd(X, y, alpha=0.5, n_iters=1000)
    with sketchy():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.9, wspace=0.22)
        a1.plot(np.arange(1, len(J) + 1), J, color=C["blue"], lw=3)
        a1.set_xscale("log")
        a1.set_xlim(1, 1000)
        a1.set_xlabel("iteration (log scale)")
        a1.set_ylabel("$J(\\mathbf{w}, b)$")
        a1.set_title("learning curve, $\\alpha = 0.5$", fontsize=16)
        despine(a1)
        g1, g2 = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-3.0, 3.2, 300))
        F = sigmoid(w[0] * g1 + w[1] * g2 + b)
        a2.contourf(g1, g2, (F >= 0.5).astype(float), levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]],
                    alpha=0.1)
        cs = a2.contour(g1, g2, F, levels=[0.1, 0.5, 0.9], colors=[C["grey"], C["green"], C["grey"]],
                        linewidths=[1.6, 3, 1.6], linestyles=["--", "-", "--"])
        a2.clabel(cs, fmt={0.1: "f = 0.1", 0.5: "f = 0.5", 0.9: "f = 0.9"}, fontsize=12)
        neg_points(a2, X[y == 0, 0], X[y == 0, 1], s=50)
        pos_points(a2, X[y == 1, 0], X[y == 1, 1], s=50)
        a2.set_xlim(-3.2, 3.2)
        a2.set_ylim(-3.0, 3.2)
        a2.set_aspect("equal")
        a2.set_xlabel("$x_1$")
        a2.set_ylabel("$x_2$")
        a2.set_title("trained model after 1000 iterations", fontsize=16)
        despine(a2)
        save(fig, "gd_training")
    print(f"    GD: w = {np.round(w, 3)}, b = {b:.3f}, final J = {J[-1]:.4f}")
    return w, b


def fig_sklearn_breast_cancer():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, names = load_breast_cancer_xy()
    X = X[:, :2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.9)
        _region_plot(ax, lambda a, b: model.decision_function(np.c_[a.ravel(), b.ravel()]).reshape(a.shape),
                     (6, 29), (9, 40), alpha=0.1)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], marker="o", s=40, facecolor="none",
                   edgecolor=C["blue"], linewidths=1.6, label="benign (test)")
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], marker="x", s=40, color=C["red"],
                   linewidths=1.8, label="malignant (test)")
        ax.set_xlim(6, 29)
        ax.set_ylim(9, 40)
        ax.set_xlabel("mean radius")
        ax.set_ylabel("mean texture")
        ax.set_title(f"test accuracy {acc:.1%}", fontsize=16)
        ax.legend(loc="upper right", frameon=True, fontsize=13)
        despine(ax)
        save(fig, "sklearn_breast_cancer")
    print(f"    breast cancer: test accuracy {acc:.4f}")


# =========================================================================== part 3b
# --------------------------------------------------------------------------- helpers
from sklearn.datasets import load_iris, make_blobs, make_moons  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

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


def softmax(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


TOY_X = np.array([[0.5, 1.5], [1, 1.2], [1.5, 1.3], [3, 0.5], [2, 4], [1, 2.4]])
TOY_Y = np.array([0, 0, 1, 1, 2, 2])


def iris_split():
    iris = load_iris()
    X, y = iris.data[:, 2:], iris.target                 # petal length, petal width
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y), list(iris.target_names)


# --------------------------------------------------------------------------- figures
def fig_binary_vs_multiclass():
    Xb, yb = make_blobs(n_samples=[25, 25], centers=[[-1.2, 1.0], [1.2, -0.8]], cluster_std=0.65, random_state=1)
    Xm, ym = make_blobs(n_samples=[20, 20, 20], centers=[[-1.5, 1.2], [1.6, 1.3], [0.0, -1.4]], cluster_std=0.55,
                        random_state=2)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.88, wspace=0.12)
        lim = (-3.2, 3.2)
        for ax, X, y, title in [(axs[0], Xb, yb, "binary: $K = 2$ classes"), (axs[1], Xm, ym, "multiclass: $K = 3$ classes")]:
            model = LogisticRegression().fit(X, y)
            K = len(np.unique(y))
            region_plot(ax, model.predict, lim, lim, K)
            class_points(ax, X, y, s=55)
            plain_axes(ax, lim, lim)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=16)
        save(fig, "binary_vs_multiclass")


def fig_probabilities():
    p = np.array([0.12, 0.30, 0.20, 0.18, 0.08, 0.12])
    k = np.arange(1, 7)
    with sketchy():
        fig, ax = plt.subplots(figsize=(8, 3.8))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.93)
        cols = [C["light"]] * 6
        cols[1] = C["red"]
        ax.bar(k, p, color=cols, edgecolor="k", lw=1.5, width=0.7)
        for kk, pp in zip(k, p):
            ax.text(kk, pp + 0.012, f"{pp:.0%}", ha="center", fontsize=14)
        ax.annotate("largest probability: $\\hat y = 2$", xy=(2, 0.31), xytext=(3.2, 0.31), fontsize=15,
                    color=C["red"], va="center", arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.6))
        ax.text(6.4, 0.25, "sum = 100%", fontsize=15, ha="right")
        ax.set_xticks(k)
        ax.set_xticklabels([f"$k = {kk}$" for kk in k])
        ax.set_ylabel("$P(y = k \\mid \\mathbf{x})$")
        ax.set_ylim(0, 0.38)
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        despine(ax)
        save(fig, "probabilities")


def fig_softmax():
    z = np.array([2.0, 1.0, 0.1])
    e = np.exp(z)
    p = e / e.sum()
    k = np.arange(1, 4)
    with sketchy():
        fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.16, top=0.85, wspace=0.35)
        for ax, v, title, fmt, col in [(axs[0], z, "scores $z_k$", "{:.1f}", C["blue"]),
                                       (axs[1], e, "$e^{z_k}$: all positive", "{:.2f}", C["orange"]),
                                       (axs[2], p, "$e^{z_k} / \\sum_l e^{z_l}$: sum to 1", "{:.3f}", C["green"])]:
            ax.bar(k, v, color=col, edgecolor="k", lw=1.5, width=0.6)
            for kk, vv in zip(k, v):
                ax.text(kk, vv + 0.03 * v.max(), fmt.format(vv), ha="center", fontsize=14)
            ax.set_xticks(k)
            ax.set_xticklabels([f"$k = {kk}$" for kk in k])
            ax.set_yticks([])
            ax.set_ylim(0, 1.2 * v.max())
            ax.set_title(title, fontsize=16)
            despine(ax, keep=("bottom",))
        for x0 in (0.355, 0.69):
            fig.text(x0, 0.5, "$\\Rightarrow$", fontsize=26, ha="center", va="center")
        save(fig, "softmax_example")


def fig_toy():
    model = LogisticRegression().fit(TOY_X, TOY_Y)
    lim_x, lim_y = (0.2, 3.3), (0.2, 4.3)
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(5.6, 5.0))
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.13, top=0.97)
            if step == 2:
                region_plot(ax, model.predict, lim_x, lim_y, 3)
            class_points(ax, TOY_X, TOY_Y, s=110, labels=["class 0", "class 1", "class 2"])
            plain_axes(ax, lim_x, lim_y, ticks=True)
            ax.set_xticks([1, 2, 3])
            ax.set_yticks([1, 2, 3, 4])
            ax.legend(loc="upper right", fontsize=13, frameon=True)
            save(fig, f"toy_step{step}")


def fig_iris():
    (X_tr, X_te, y_tr, y_te), names = iris_split()
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    lim_x, lim_y = (0.6, 7.3), (-0.1, 2.75)
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.6, 4.9))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.9)
        region_plot(ax, model.predict, lim_x, lim_y, 3)
        class_points(ax, X_te, y_te, s=60, labels=names)
        plain_axes(ax, lim_x, lim_y, "petal length (cm)", "petal width (cm)", ticks=True)
        ax.set_title(f"softmax regression on the iris data: test accuracy {acc:.1%}", fontsize=15)
        ax.legend(loc="upper left", fontsize=13, frameon=True)
        save(fig, "iris_regions")
        cm = confusion_matrix(y_te, model.predict(X_te))
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        fig.subplots_adjust(left=0.3, right=0.97, bottom=0.25, top=0.95)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=2.2 * cm.max())
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22, color="k")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(names, rotation=20)
        ax.set_yticklabels(names)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        save(fig, "iris_confusion")
    print(f"    iris test accuracy {acc:.4f}; confusion matrix\n{cm}")


def fig_one_vs_rest():
    (X_tr, _, y_tr, _), names = iris_split()
    lim_x, lim_y = (0.6, 7.3), (-0.1, 2.75)
    with sketchy():
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.1, top=0.88, wspace=0.1)
        for k, ax in enumerate(axs):
            yk = (y_tr == k).astype(int)
            model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, yk)
            g1, g2 = np.meshgrid(np.linspace(*lim_x, 300), np.linspace(*lim_y, 300))
            F = model.predict_proba(np.c_[g1.ravel(), g2.ravel()])[:, 1].reshape(g1.shape)
            ax.contourf(g1, g2, (F >= 0.5).astype(float), levels=[-0.5, 0.5, 1.5], colors=[C["light"], CLASS_COLORS[k]],
                        alpha=0.15)
            ax.contour(g1, g2, F, levels=[0.5], colors="k", linewidths=2.2)
            rest = y_tr != k
            ax.scatter(X_tr[rest, 0], X_tr[rest, 1], marker=".", s=40, color=C["grey"], zorder=4)
            class_points(ax, X_tr[~rest], y_tr[~rest], s=45)
            plain_axes(ax, lim_x, lim_y, "petal length", "petal width")
            ax.set_title(f"{names[k]} vs rest", fontsize=16, color=CLASS_COLORS[k])
        save(fig, "one_vs_rest")


FIGURES = {   # name: (output folder, function)
    "linreg_fail": (OUT_A, fig_linreg_fail),
    "sigmoid": (OUT_A, fig_sigmoid),
    "model_diagram": (OUT_A, fig_model_diagram),
    "logistic_fit": (OUT_A, fig_logistic_fit),
    "sigmoid_threshold": (OUT_A, fig_sigmoid_threshold),
    "threshold_radius": (OUT_A, fig_threshold_radius),
    "boundary_linear": (OUT_A, fig_boundary_linear),
    "boundary_ellipse": (OUT_A, fig_boundary_ellipse),
    "boundary_wiggly": (OUT_A, fig_boundary_wiggly),
    "cost_surfaces": (OUT_A, fig_cost_surfaces),
    "loss_curves": (OUT_A, fig_loss_curves),
    "gd_training": (OUT_A, fig_gd_training),
    "sklearn": (OUT_A, fig_sklearn_breast_cancer),
    "binary_vs_multiclass": (OUT_B, fig_binary_vs_multiclass),
    "probabilities": (OUT_B, fig_probabilities),
    "softmax": (OUT_B, fig_softmax),
    "toy": (OUT_B, fig_toy),
    "iris": (OUT_B, fig_iris),
    "one_vs_rest": (OUT_B, fig_one_vs_rest),
}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}")
    for key, (folder, fn) in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            OUT = folder                    # save() writes to the folder of this figure
            print(f"{key}  ->  {folder.relative_to(ROOT)}")
            fn()
