"""
Sketch-style figures for Lecture 4: Overfitting, regularisation and model selection.

Run from anywhere:
    python scripts/make_figures_lecture04.py                  # all figures
    python scripts/make_figures_lecture04.py fit lambda       # only figures whose name contains these words

Output: figures/overfitting-sketch/*.png (relative to the project root).

Requirements: numpy, matplotlib >= 3.7, scikit-learn. All data are synthetic and generated here with
fixed random seeds, so the figures (and the numbers quoted on the slides) are reproducible.

Fonts: the hand-drawn look uses the "xkcd Script" font (scripts/fonts/xkcd-script.ttf,
CC BY-NC 3.0, https://github.com/ipython/xkcd-font). If it is missing, Matplotlib falls back to
Humor Sans / Comic Neue / Comic Sans MS, so the script still runs.

Figures that are shown one after another on the same slide ("builds") are drawn on canvases of
identical size with fixed margins (no bbox_inches="tight"), so they line up exactly.

Regularised fits use the cost of the lecture,
    J(w, b) = 1/(2m) * sum (f(x_i) - y_i)^2 + lambda/(2m) * sum_j w_j^2,
which has the same minimiser as scikit-learn's Ridge(alpha=lambda). Polynomial features are always
standardised before fitting.
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
OUT = ROOT / "figures" / "overfitting-sketch"
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


def neg_points(ax, x, y, s=110, **kw):
    """Negative class (y = 0): open circles."""
    ax.scatter(x, y, marker="o", s=s, facecolor="white", edgecolor=C["blue"], linewidths=2.2, zorder=5, **kw)


def pos_points(ax, x, y, s=110, **kw):
    """Positive class (y = 1): crosses."""
    ax.scatter(x, y, marker="x", s=s, color=C["red"], linewidths=2.6, zorder=5, **kw)


LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


def _region_plot(ax, decision, xlim, ylim, alpha=0.1):
    g1, g2 = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = decision(g1, g2)
    ax.contourf(g1, g2, (Z >= 0).astype(float), levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]],
                alpha=alpha)
    ax.contour(g1, g2, Z, levels=[0], colors=[C["green"]], linewidths=3)


# --------------------------------------------------------------------------- data and helpers
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402


def true_f(x):
    """The (unknown) function behind the regression data."""
    return 1.0 + 2.2 * x - 1.6 * x ** 2


NOISE = 0.1


def make_regression_data(m, seed, grid=False):
    rng = np.random.default_rng(seed)
    if grid:
        x = np.linspace(0.05, 1.0, m) + rng.uniform(-0.02, 0.02, m)
    else:
        x = np.sort(rng.uniform(0.02, 1.0, m))
    return x, true_f(x) + rng.normal(0, NOISE, m)


X_TRAIN, Y_TRAIN = make_regression_data(11, seed=5, grid=True)      # the small training set of the slides


def poly_model(degree, lam=0.0):
    """Polynomial features -> standardisation -> (regularised) least squares.

    lam is the lambda of the lecture; Ridge(alpha=lam) minimises the same cost (up to the factor 2m)."""
    reg = LinearRegression() if lam == 0 else Ridge(alpha=lam)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)


def fit(degree, x, y, lam=0.0):
    return poly_model(degree, lam).fit(x.reshape(-1, 1), y)


def predict(model, x):
    return model.predict(np.asarray(x).reshape(-1, 1))


def mse(model, x, y):
    return float(np.mean((predict(model, x) - y) ** 2))


XS = np.linspace(0.0, 1.05, 500)
YLIM = (0.6, 2.25)


def reg_axes(ax, title=None, color="k"):
    ax.set_xlim(0, 1.05)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if title:
        ax.set_title(title, fontsize=16, color=color)
    despine(ax)


def plot_fit(ax, model, color=C["blue"], lw=3, **kw):
    ax.plot(XS, np.clip(predict(model, XS), YLIM[0] - 1, YLIM[1] + 1), color=color, lw=lw, **kw)


# --------------------------------------------------------------------------- regression figures
def fig_data():
    with sketchy():
        fig, ax = plt.subplots(figsize=(7, 4.4))
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.96)
        data_points(ax, X_TRAIN, Y_TRAIN, s=110)
        reg_axes(ax)
        save(fig, "data")


def fig_fit_regression():
    """Underfit (degree 1), overfit (degree 10), just right (degree 2); cumulative three-step build."""
    panels = [(1, "underfit: degree 1", C["red"]), (2, "just right: degree 2", C["green"]),
              (10, "overfit: degree 10", C["red"])]
    order = [0, 2, 1]                      # reveal order: underfit, overfit, just right
    for step in (1, 2, 3):
        shown = set(order[:step])
        with sketchy():
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
            fig.subplots_adjust(left=0.03, right=0.99, bottom=0.12, top=0.88, wspace=0.12)
            for k, (ax, (deg, title, col)) in enumerate(zip(axs, panels)):
                if k not in shown:
                    ax.axis("off")
                    continue
                model = fit(deg, X_TRAIN, Y_TRAIN)
                plot_fit(ax, model)
                data_points(ax, X_TRAIN, Y_TRAIN, s=70)
                reg_axes(ax, title, col)
            save(fig, f"fit_regression_step{step}")


def fig_bias_variance():
    """Fits on six different training sets: degree 1 is consistently wrong, degree 10 is all over the place."""
    sets = [make_regression_data(11, seed=s, grid=True) for s in range(20, 26)]
    cmap = plt.get_cmap("viridis")
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.12, top=0.86, wspace=0.12)
        for ax, deg, title in [(axs[0], 1, "degree 1 on 6 training sets: high bias"),
                               (axs[1], 10, "degree 10 on 6 training sets: high variance")]:
            for k, (x, y) in enumerate(sets):
                plot_fit(ax, fit(deg, x, y), color=cmap(0.85 * k / 5), lw=2)
            ax.plot(XS, true_f(XS), color="k", lw=2.4, ls="--")
            reg_axes(ax, title)
        axs[0].text(0.62, 0.75, "dashed: true function", fontsize=13)
        save(fig, "bias_variance")


def fig_more_data():
    x_big, y_big = make_regression_data(120, seed=9)
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.12, top=0.86, wspace=0.12)
        for ax, (x, y), title in [(axs[0], (X_TRAIN, Y_TRAIN), "degree 10, $m = 11$: overfit"),
                                  (axs[1], (x_big, y_big), "degree 10, $m = 120$: fits well")]:
            plot_fit(ax, fit(10, x, y))
            data_points(ax, x, y, s=70 if len(x) < 50 else 30)
            reg_axes(ax, title)
        save(fig, "more_data")


LAM_GOOD = 0.05


def fig_regularised_fit():
    """Degree 10 without and with regularisation, plus the size of the weights."""
    m0 = fit(10, X_TRAIN, Y_TRAIN)
    m1 = fit(10, X_TRAIN, Y_TRAIN, lam=LAM_GOOD)
    w0, w1 = m0[-1].coef_, m1[-1].coef_
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.12, top=0.86, wspace=0.12)
        for ax, model, title, col in [(axs[0], m0, "degree 10, $\\lambda = 0$", C["red"]),
                                      (axs[1], m1, f"degree 10, $\\lambda = {LAM_GOOD:g}$", C["green"])]:
            plot_fit(ax, model)
            data_points(ax, X_TRAIN, Y_TRAIN, s=70)
            reg_axes(ax, title, col)
        save(fig, "regularised_fit")
        fig, ax = plt.subplots(figsize=(11, 3.6))
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.18, top=0.9)
        j = np.arange(1, 11)
        ax.bar(j - 0.2, np.abs(w0), width=0.38, color=C["red"], label="$\\lambda = 0$")
        ax.bar(j + 0.2, np.abs(w1), width=0.38, color=C["green"], label=f"$\\lambda = {LAM_GOOD:g}$")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e9)
        ax.set_xticks(j)
        ax.set_xticklabels([f"$w_{{{k}}}$" for k in j])
        ax.set_ylabel("$|w_j|$ (log scale)")
        ax.legend(loc="upper left", frameon=False, fontsize=14, ncol=2)
        despine(ax)
        save(fig, "regularised_weights")
    print(f"    |w| max: lambda=0 {np.abs(w0).max():.3g}, lambda={LAM_GOOD} {np.abs(w1).max():.3g}")
    return w0, w1


def fig_lambda_sweep():
    lams = [(0.0, "$\\lambda = 0$: overfit", C["red"]),
            (LAM_GOOD, f"$\\lambda = {LAM_GOOD:g}$: just right", C["green"]),
            (1e4, "$\\lambda = 10^4$: underfit", C["red"])]
    with sketchy():
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.12, top=0.88, wspace=0.12)
        for ax, (lam, title, col) in zip(axs, lams):
            plot_fit(ax, fit(10, X_TRAIN, Y_TRAIN, lam))
            data_points(ax, X_TRAIN, Y_TRAIN, s=70)
            reg_axes(ax, title, col)
        axs[2].axhline(Y_TRAIN.mean(), color="k", ls=":", lw=1.4)
        axs[2].text(0.45, Y_TRAIN.mean() - 0.16, "$f(x) \\approx b = \\bar y$", fontsize=15)
        save(fig, "lambda_sweep")
    big = fit(10, X_TRAIN, Y_TRAIN, 1e4)[-1]
    print(f"    lambda=1e4: max|w| = {np.abs(big.coef_).max():.4f}, b = {big.intercept_:.4f}, mean y = {Y_TRAIN.mean():.4f}")


def fig_validation_curves():
    x_tr, y_tr = make_regression_data(15, seed=31)
    x_va, y_va = make_regression_data(400, seed=32)
    degrees = np.arange(1, 13)
    tr = [mse(fit(d, x_tr, y_tr), x_tr, y_tr) for d in degrees]
    va = [mse(fit(d, x_tr, y_tr), x_va, y_va) for d in degrees]
    lams = np.logspace(-10, 2, 49)                       # lambda curve: the 11-point training set of the slides
    trl = [mse(fit(10, X_TRAIN, Y_TRAIN, l), X_TRAIN, Y_TRAIN) for l in lams]
    val = [mse(fit(10, X_TRAIN, Y_TRAIN, l), x_va, y_va) for l in lams]
    best_d = int(degrees[int(np.argmin(va))])
    best_l = float(lams[int(np.argmin(val))])
    with sketchy():
        for name, xs, a, bv, xlabel, logx, best in [
            ("validation_degree", degrees, tr, va, "polynomial degree $d$", False, best_d),
            ("validation_lambda", lams, trl, val, "$\\lambda$ (log scale), degree 10, $m = 11$", True, best_l),
        ]:
            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.95)
            ax.plot(xs, a, color=C["blue"], lw=2.8, marker="o" if not logx else None, ms=5, label="training error")
            ax.plot(xs, bv, color=C["red"], lw=2.8, marker="o" if not logx else None, ms=5, label="validation error")
            ax.axvline(best, color=C["green"], ls="--", lw=2)
            ax.axhline(NOISE ** 2, color="k", ls=":", lw=1.4)
            ax.set_yscale("log")
            ax.set_ylim(5e-4 if logx else 1e-4, 0.3 if logx else 3)
            if logx:
                ax.set_xscale("log")
            else:
                ax.set_xticks(degrees)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("MSE (log scale)")
            ax.legend(loc="upper center" if logx else "upper right", frameon=False, fontsize=13)
            despine(ax)
            save(fig, name)
    print(f"    best degree {best_d}, best lambda {best_l:.3g}; val MSE at best degree {min(va):.4f}, noise var {NOISE ** 2}")
    print(f"    degree 1: train {tr[0]:.4f} val {va[0]:.4f}; degree 10: train {tr[9]:.2e} val {va[9]:.3f}")
    i = int(np.argmin(val))
    print(f"    lambda curve: lambda->0 val {val[0]:.3g}, best val {val[i]:.4f} at {lams[i]:.3g}, lambda=1e-3 val "
          f"{mse(fit(10, X_TRAIN, Y_TRAIN, 1e-3), x_va, y_va):.4f}, lambda=1e2 val {val[-1]:.4f}")


# --------------------------------------------------------------------------- classification figures
def make_classification_data(m=70, seed=3, flip=0.08):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(m, 2))
    y = ((X[:, 0] + 0.3) ** 2 + 1.4 * (X[:, 1] + 0.25) ** 2 > 0.45).astype(int)
    noisy = rng.random(m) < flip
    y[noisy] = 1 - y[noisy]
    return X, y


def class_model(degree, C_=1e6):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(),
                         LogisticRegression(C=C_, max_iter=20000))


def _class_panel(ax, model, X, y, title, col):
    _region_plot(ax, lambda a, b: model.decision_function(np.c_[a.ravel(), b.ravel()]).reshape(a.shape),
                 (-1.05, 1.05), (-1.05, 1.05), alpha=0.1)
    neg_points(ax, X[y == 0, 0], X[y == 0, 1], s=55)
    pos_points(ax, X[y == 1, 0], X[y == 1, 1], s=55)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title, fontsize=16, color=col)
    despine(ax)


def fig_fit_classification():
    X, y = make_classification_data()
    panels = [(1, "underfit: degree 1", C["red"]), (2, "just right: degree 2", C["green"]),
              (8, "overfit: degree 8", C["red"])]
    models = [class_model(d).fit(X, y) for d, _, _ in panels]
    order = [0, 2, 1]
    for step in (1, 2, 3):
        shown = set(order[:step])
        with sketchy():
            fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
            fig.subplots_adjust(left=0.03, right=0.99, bottom=0.1, top=0.9, wspace=0.12)
            for k, (ax, (deg, title, col), model) in enumerate(zip(axs, panels, models)):
                if k not in shown:
                    ax.axis("off")
                    continue
                _class_panel(ax, model, X, y, title, col)
            save(fig, f"fit_classification_step{step}")
    print("    training accuracy by degree:", [f"{m.score(X, y):.2f}" for m in models])


def fig_regularised_classification():
    X, y = make_classification_data()
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.9, wspace=0.12)
        for ax, lam, title, col in [(axs[0], 1e-6, "degree 8, $\\lambda \\approx 0$", C["red"]),
                                    (axs[1], 1.0, "degree 8, $\\lambda = 1$", C["green"])]:
            _class_panel(ax, class_model(8, C_=1.0 / lam).fit(X, y), X, y, title, col)
        save(fig, "regularised_classification")


FIGURES = {
    "data": fig_data,
    "fit_regression": fig_fit_regression,
    "bias_variance": fig_bias_variance,
    "more_data": fig_more_data,
    "regularised_fit": fig_regularised_fit,
    "lambda_sweep": fig_lambda_sweep,
    "validation": fig_validation_curves,
    "fit_classification": fig_fit_classification,
    "regularised_classification": fig_regularised_classification,
}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}  ->  {OUT}")
    for key, fn in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            print(key)
            fn()
