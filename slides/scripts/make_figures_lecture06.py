"""
Sketch-style figures for Lecture 6: PyTorch.

Run from anywhere:
    python scripts/make_figures_lecture06.py               # all figures
    python scripts/make_figures_lecture06.py graph loop    # only figures whose name contains these words

Output: figures/pytorch-sketch/*.png (relative to the project root).

Requirements: numpy, matplotlib >= 3.7, scikit-learn; PyTorch for the two learning-curve figures
(the diagrams are drawn without PyTorch, so they can be regenerated even where it is not installed).

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
OUT = ROOT / "figures" / "pytorch-sketch"
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
def box(ax, xc, yc, text, w, h, fc="#FCE4EC", ec="k", fs=15, lw=2, color="k", round_=0.15):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle=f"round,pad=0.04,rounding_size={round_}", fc=fc, ec=ec, lw=lw))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, p, q, color="k", lw=2.0, ms=18, style="-|>", rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=lw, color=color, mutation_scale=ms,
                                                    connectionstyle=f"arc3,rad={rad}"))


def blank_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def load_scaled_diabetes():
    """The ten diabetes features of Lecture 2b, standardised, as float32 tensors."""
    import torch
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler

    d = load_diabetes(scaled=False)
    X = torch.tensor(StandardScaler().fit_transform(d.data), dtype=torch.float32)
    y = torch.tensor(d.target, dtype=torch.float32).reshape(-1, 1)
    return X, y


def least_squares_mse(X, y):
    """MSE of the exact least-squares fit (computed with torch.linalg.lstsq in double precision)."""
    import torch

    A = torch.cat([torch.ones(len(y), 1), X], dim=1).double()
    theta = torch.linalg.lstsq(A, y.double()).solution
    return float(((A @ theta - y.double()) ** 2).mean())


# --------------------------------------------------------------------------- figures
def fig_tensor_ranks():
    with sketchy():
        fig, ax = plt.subplots(figsize=(12.5, 3.9))
        fig.subplots_adjust(0, 0, 1, 1)
        blank_axes(ax, (0, 16.5), (-0.2, 5.0))
        cell = 0.9

        def grid(x0, y0, rows, cols, fc, vals=None):
            for r in range(rows):
                for c in range(cols):
                    ax.add_patch(plt.Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell, fc=fc, ec="k", lw=1.8))
                    if vals is not None:
                        ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, str(vals[r][c]), ha="center",
                                va="center", fontsize=17)

        grid(1.0, 3.0, 1, 1, "#FFF8E1", [[7]])
        grid(3.2, 3.0, 1, 3, "#E3F2FD", [[1, 2, 3]])
        grid(7.3, 3.45, 2, 3, "#E8F5E9", [[1, 2, 3], [4, 5, 6]])
        for k, fc in enumerate(["#FFF8E1", "#FCE4EC", "#F3E5F5"]):
            grid(11.6 + 0.45 * k, 3.45 + 0.45 * k, 2, 3, fc)
        for xc, t, sh in [(1.45, "scalar", "shape ()"), (4.55, "vector", "shape (3,)"), (8.65, "matrix", "shape (2, 3)"),
                          (13.5, "3D tensor", "shape (3, 2, 3)")]:
            ax.text(xc, 4.65, t, ha="center", fontsize=18)
            ax.text(xc, 0.75, sh, ha="center", fontsize=16, color=C["navy"])
        ax.text(13.5, 0.05, "e.g. an image: 3 colour channels of 2 x 3 pixels", ha="center", fontsize=13, color=C["grey"])
        save(fig, "tensor_ranks")


def fig_graph():
    """Computational graph for L = (w x + b - y)^2 with w = 2, x = 3, b = 1, y = 5."""
    leaves = {"w": (1.0, 4.2, "$w = 2$", "$\\partial L/\\partial w = 12$"),
              "x": (1.0, 3.0, "$x = 3$", None),
              "b": (1.0, 1.8, "$b = 1$", "$\\partial L/\\partial b = 4$"),
              "y": (1.0, 0.6, "$y = 5$", None)}
    ops = {"mul": (4.0, 3.6, "$wx$", "6", "4"),
           "add": (6.9, 2.7, "$z = wx + b$", "7", "4"),
           "sub": (9.6, 1.8, "$z - y$", "2", "4"),
           "sq": (12.3, 1.8, "$L = (z - y)^2$", "4", "1")}
    pos = {k: v[:2] for k, v in {**leaves, **ops}.items()}
    widths = {"w": 1.4, "x": 1.4, "b": 1.4, "y": 1.4, "mul": 1.2, "add": 2.1, "sub": 1.5, "sq": 2.3}
    edges = [("w", "mul"), ("x", "mul"), ("mul", "add"), ("b", "add"), ("add", "sub"), ("y", "sub"), ("sub", "sq")]
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(12, 4.9))
            fig.subplots_adjust(0, 0, 1, 1)
            blank_axes(ax, (0, 13.8), (-0.2, 5.4))
            for a, b_ in edges:
                (xa, ya), (xb, yb) = pos[a], pos[b_]
                arrow(ax, (xa + widths[a] / 2 + 0.05, ya), (xb - widths[b_] / 2 - 0.08, yb), lw=1.8)
            for key, (xc, yc, lab, grad) in leaves.items():
                track = key in ("w", "b")
                box(ax, xc, yc, lab, widths[key], 0.62, fc="#E3F2FD" if track else "#F5F5F5",
                    ec=C["blue"] if track else "k", fs=16)
                if step == 2 and grad:
                    ax.text(xc, yc - 0.42, grad, ha="center", va="top", fontsize=13, color=C["red"])
            for key, (xc, yc, lab, val, grad) in ops.items():
                box(ax, xc, yc, lab, widths[key], 0.62, fc="#FFF8E1", fs=16)
                ax.text(xc, yc + 0.42, f"value {val}", ha="center", va="bottom", fontsize=13, color=C["navy"])
                if step == 2:
                    ax.text(xc, yc - 0.42, f"$\\partial L/\\partial(\\cdot) = {grad}$", ha="center", va="top",
                            fontsize=13, color=C["red"])
            ax.text(7.2, 5.05, "forward pass: compute the values  $\\longrightarrow$", fontsize=15, color=C["navy"],
                    ha="center")
            if step == 2:
                ax.text(7.2, 4.55, "$\\longleftarrow$  backward pass: the chain rule gives every $\\partial L/\\partial(\\cdot)$",
                        fontsize=15, color=C["red"], ha="center")
            ax.text(0.3, 5.05, "blue: requires_grad=True", fontsize=12, color=C["blue"])
            save(fig, f"graph_step{step}")


def fig_training_loop():
    steps = [("1. forward pass", "y_hat = model(X)"), ("2. loss", "loss = loss_fn(y_hat, y)"),
             ("3. reset gradients", "optimizer.zero_grad()"), ("4. backward pass", "loss.backward()"),
             ("5. update parameters", "optimizer.step()")]
    with sketchy():
        fig, ax = plt.subplots(figsize=(10, 5.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank_axes(ax, (-5.5, 5.5), (-3.2, 3.2))
        R = 2.35
        angles = np.radians(90 - 72 * np.arange(5))
        pos = [(1.65 * R * np.cos(a), R * np.sin(a)) for a in angles]
        cols = ["#E3F2FD", "#FFF8E1", "#F5F5F5", "#FCE4EC", "#E8F5E9"]
        for (xc, yc), (t, code), fc in zip(pos, steps, cols):
            box(ax, xc, yc, f"{t}\n{code}", 3.4, 0.95, fc=fc, fs=13)
        for k in range(5):
            (x0, y0), (x1, y1) = pos[k], pos[(k + 1) % 5]
            d = np.array([x1 - x0, y1 - y0])
            u = d / np.linalg.norm(d)
            arrow(ax, (x0 + 0.95 * u[0] * 1.6, y0 + 0.62 * u[1] * 1.6), (x1 - 0.95 * u[0] * 1.6, y1 - 0.62 * u[1] * 1.6),
                  lw=2.2, rad=-0.15)
        ax.text(0, 0.15, "repeat for every\nepoch (and mini-batch)", ha="center", va="center", fontsize=15,
                color=C["navy"])
        save(fig, "training_loop")


def fig_learning_curve():
    """Full-batch training of nn.Linear on the scaled diabetes data (the code on the slides)."""
    import torch
    from torch import nn

    X, y = load_scaled_diabetes()
    torch.manual_seed(0)
    model = nn.Linear(10, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range(500):
        loss = loss_fn(model(X), y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    best = least_squares_mse(X, y)
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.6, 4.5))
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.95)
        ax.plot(np.arange(1, 501), losses, color=C["blue"], lw=3)
        ax.axhline(best, color="k", ls=":", lw=1.5)
        ax.text(1.2, best + 1200, "least-squares optimum", ha="left", fontsize=13)
        ax.set_xscale("log")
        ax.set_xlim(1, 500)
        ax.set_ylim(0, 31000)
        ax.set_xlabel("epoch (log scale)")
        ax.set_ylabel("MSE loss")
        despine(ax)
        save(fig, "learning_curve")
    print(f"    final loss {losses[-1]:.1f}, least-squares MSE {best:.1f}")


def fig_minibatch():
    """Loss on the whole data set after each epoch, for three batch sizes."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    X, y = load_scaled_diabetes()
    best = least_squares_mse(X, y)
    runs = [(len(y), 0.1, "full batch ($m = 442$): 1 step per epoch", C["blue"]),
            (32, 0.05, "mini-batches of 32: 14 steps per epoch", C["green"]),
            (1, 0.002, "single examples: 442 steps per epoch", C["red"])]
    curves = []
    for bs, lr, label, col in runs:
        torch.manual_seed(0)
        model = nn.Linear(10, 1)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)
        hist = []
        with torch.no_grad():
            hist.append(loss_fn(model(X), y).item())
        for _ in range(30):
            for Xb, yb in loader:
                loss = loss_fn(model(Xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                hist.append(loss_fn(model(X), y).item())
        curves.append((hist, label, col, lr))
    with sketchy():
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        for hist, label, col, lr in curves:
            ax.plot(np.arange(len(hist)), hist, color=col, lw=2.8, marker="o", ms=3.5,
                    label=f"{label}, lr = {lr:g}")
        ax.axhline(best, color="k", ls=":", lw=1.5)
        ax.set_yscale("log")
        ax.set_yticks([3000, 5000, 10000, 20000])
        ax.set_yticklabels(["3000", "5000", "10000", "20000"])
        ax.minorticks_off()
        ax.set_xlabel("epoch (one pass over the training data)")
        ax.set_ylabel("MSE on all data (log scale)")
        ax.legend(loc="upper right", fontsize=12, frameon=False)
        despine(ax)
        save(fig, "minibatch_curves")
    for hist, label, _, _ in curves:
        print(f"    {label}: after 1 epoch {hist[1]:.0f}, after 30 epochs {hist[-1]:.1f}")


# --------------------------------------------------------------------------- course infrastructure
def fig_hpc_workflow():
    """Working on the university's GPU cluster: laptop -> login node -> GPU nodes, with a sketched terminal."""
    lines = [("$ ssh <your-guid>@<login-node>", "#E8EDF3"),
             ("$ cd ml-course", "#E8EDF3"),
             ("$ python cnn_digits.py", "#E8EDF3"),
             ("epoch  1/10   loss 0.312", "#9FD8B4"),
             ("epoch  2/10   loss 0.087", "#9FD8B4"),
             ("   ...", "#9FD8B4"),
             ("epoch 10/10   loss 0.011", "#9FD8B4"),
             ("test accuracy: 0.99", "#F2C26B"),
             ("$ exit", "#E8EDF3")]
    with sketchy():
        fig = plt.figure(figsize=(12.5, 5.0))
        ax = fig.add_axes([0.0, 0.0, 0.52, 1.0])
        blank_axes(ax, (0, 10.5), (0, 8))
        ax.add_patch(FancyBboxPatch((4.1, 0.6), 6.1, 6.4, boxstyle="round,pad=0.05,rounding_size=0.3",
                                    fc="#FAFAFA", ec=C["grey"], lw=1.6, ls="--"))
        ax.text(7.15, 7.25, "HPC cluster", ha="center", fontsize=16, color=C["grey"])
        box(ax, 1.7, 5.3, "your laptop\ncode, data", 2.8, 1.5, fc="#E3F2FD", ec=C["blue"], fs=15)
        box(ax, 7.15, 5.3, "login node\nconnect, edit, start", 3.7, 1.5, fc="#FFF8E1", ec=C["orange"], fs=15)
        box(ax, 7.15, 1.9, "GPU nodes\ntrain the models", 3.7, 1.5, fc="#E8F5E9", ec=C["green"], fs=15)
        arrow(ax, (3.15, 5.55), (5.25, 5.55))
        ax.text(4.2, 5.8, "ssh", ha="center", fontsize=14)
        arrow(ax, (5.25, 5.0), (3.15, 5.0))
        ax.text(4.2, 4.45, "results", ha="center", fontsize=13, color=C["grey"])
        arrow(ax, (6.7, 4.5), (6.7, 2.7))
        arrow(ax, (7.6, 2.7), (7.6, 4.5))
        ax.text(8.7, 3.6, "run\njob", ha="center", va="center", fontsize=13, color=C["grey"])
        tx = fig.add_axes([0.54, 0.08, 0.44, 0.84])
        tx.set_xlim(0, 1)
        tx.set_ylim(0, 1)
        tx.axis("off")
        tx.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.03",
                                    fc="#1E2733", ec="k", lw=2, transform=tx.transAxes))
        tx.add_patch(plt.Rectangle((0.0, 0.9), 1.0, 0.1, fc="#3A4656", ec="none", transform=tx.transAxes))
        for k, col in enumerate(["#E53E30", "#E0A100", "#00AD69"]):
            tx.add_patch(plt.Circle((0.05 + 0.045 * k, 0.95), 0.015, color=col, transform=tx.transAxes))
        # no sketchy white outline on the dark terminal: path_effects=[]
        tx.text(0.5, 0.95, "terminal (example session)", ha="center", va="center", fontsize=12, color="#C9D3DF",
                path_effects=[])
        for k, (txt, col) in enumerate(lines):
            tx.text(0.04, 0.82 - 0.088 * k, txt, fontsize=13, color=col, family="monospace", va="center",
                    path_effects=[])
        save(fig, "hpc_workflow")


def fig_course_notebooks():
    """A sketched folder of the course notebooks (the files themselves are on Moodle)."""
    names = ["quickstart", "tensors", "autograd", "datasets_dataloaders", "linear-regression",
             "logistic-regression", "logistic-regression-digits", "mnist-digits", "cnn", "sobel-filter",
             "neural-style-transfer", "parallel-pytorch"]
    with sketchy():
        fig, ax = plt.subplots(figsize=(9.6, 5.2))
        fig.subplots_adjust(0, 0, 1, 1)
        blank_axes(ax, (0, 12), (0, 6.5))
        ax.add_patch(FancyBboxPatch((0.3, 0.3), 11.4, 5.9, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc="white", ec="k", lw=2))
        ax.add_patch(plt.Rectangle((0.3, 5.45), 11.4, 0.75, fc="#E3F2FD", ec="k", lw=2))
        ax.text(0.7, 5.82, "Moodle  >  Machine Learning  >  Jupyter notebooks", fontsize=14, va="center",
                color=C["navy"])
        for k, name in enumerate(names):
            col, row = k % 3, k // 3
            x, y = 0.9 + 3.75 * col, 4.55 - 1.15 * row
            ax.add_patch(FancyBboxPatch((x, y - 0.32), 0.5, 0.64, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc="#FFE0B2", ec=C["orange"], lw=1.6))
            ax.text(x + 0.25, y, "ipynb", ha="center", va="center", fontsize=8, color=C["orange"], rotation=90)
            ax.text(x + 0.75, y, name, va="center", fontsize=13)
        save(fig, "course_notebooks")


FIGURES = {
    "tensor_ranks": fig_tensor_ranks,
    "graph": fig_graph,
    "training_loop": fig_training_loop,
    "learning_curve": fig_learning_curve,
    "minibatch": fig_minibatch,
    "hpc_workflow": fig_hpc_workflow,
    "course_notebooks": fig_course_notebooks,
}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}  ->  {OUT}")
    for key, fn in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            print(key)
            fn()
