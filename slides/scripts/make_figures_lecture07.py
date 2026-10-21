"""
Sketch-style figures for Lecture 7: Neural networks.
    Part 7a, representation and forward propagation -> figures/neural-network-sketch/
    Part 7b, backpropagation and training             -> figures/backprop-sketch/
    Part 7c, the mathematics of backpropagation       -> figures/backprop-math-sketch/
    (07_appendix_matrix_calculus.qmd uses figures/backprop-math-sketch/ as well)

Run from anywhere:
    python scripts/make_figures_lecture07.py                  # all figures (the PyTorch runs take a minute)
    python scripts/make_figures_lecture07.py network chain    # only figures whose name contains these words

Requirements: numpy, scipy, matplotlib >= 3.7, scikit-learn (8 x 8 digits, make_circles);
PyTorch for the vanishing-gradient and digit-training figures of part 7b; torchvision for the MNIST
figures of part 7a (the MNIST data, about 60 MB, are downloaded once into data/MNIST). Fixed random seeds,
so the numbers quoted on the slides are reproducible.

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
OUT_A = ROOT / "figures" / "neural-network-sketch"   # part 7a
OUT_B = ROOT / "figures" / "backprop-sketch"         # part 7b
OUT_C = ROOT / "figures" / "backprop-math-sketch"    # part 7c and the appendix
OUT = OUT_A                                          # set per figure in __main__
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
LAYER_COLORS = {"in": ("#E8F5E9", C["green"]), "hidden": ("#E3F2FD", C["blue"]), "out": ("#FCE4EC", C["red"])}
FADE = "#DDDDDD"


def box(ax, xc, yc, text, w, h, fc="#FCE4EC", ec="k", fs=15, lw=2, color="k", round_=0.15):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle=f"round,pad=0.04,rounding_size={round_}", fc=fc, ec=ec, lw=lw))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, p, q, color="k", lw=1.6, ms=14, style="-|>", rad=0.0, shrink=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=lw, color=color, mutation_scale=ms,
                                                    shrinkA=shrink, shrinkB=shrink,
                                                    connectionstyle=f"arc3,rad={rad}"))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def neuron(ax, xc, yc, r, kind="hidden", label=None, fs=14, faded=False, lw=2.0):
    fc, ec = LAYER_COLORS[kind]
    if faded:
        fc, ec = "#F7F7F7", FADE
    ax.add_patch(plt.Circle((xc, yc), r, fc=fc, ec=ec, lw=lw, zorder=4))
    if label:
        ax.text(xc, yc, label, ha="center", va="center", fontsize=fs, zorder=5, color="k" if not faded else FADE)


def connect(ax, p, q, r, color="k", lw=1.3, ms=11):
    """Arrow from circle centre p to circle centre q (both of radius r)."""
    d = np.array(q, float) - np.array(p, float)
    u = d / np.linalg.norm(d)
    arrow(ax, tuple(np.array(p) + u * r), tuple(np.array(q) - u * r), color=color, lw=lw, ms=ms)


def layer_positions(n, x, spacing, yc=0.0):
    return [(x, yc + spacing * ((n - 1) / 2 - k)) for k in range(n)]


# --------------------------------------------------------------------------- graph helpers
def node_box(ax, xc, yc, text, w, h=0.62, kind="op", fs=15):
    styles = {"leaf": ("#E3F2FD", C["blue"]), "data": ("#F5F5F5", "k"), "op": ("#FFF8E1", "k"),
              "loss": ("#FCE4EC", C["red"])}
    fc, ec = styles[kind]
    box(ax, xc, yc, text, w, h, fc=fc, ec=ec, fs=fs)


def edge(ax, p, wp, q, wq, color="k", lw=1.7, rad=0.0):
    """Arrow from the right side of box p (width wp) to the left side of box q (width wq)."""
    arrow(ax, (p[0] + wp / 2 + 0.05, p[1]), (q[0] - wq / 2 - 0.08, q[1]), color=color, lw=lw, ms=15, rad=rad)


def red(ax, x, y, text, fs=13, ha="center", va="top"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=C["red"])


# =========================================================================== part 7a
# --------------------------------------------------------------------------- figures: motivation
def fig_housing_relu():
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(2.6, 9.5, 9))
    y = 0.85 * (x - 2.2) + rng.normal(0, 0.35, x.size)
    with sketchy():
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.1, top=0.96)
        arrow_axes(ax, "size $x$", "price $y$", xlim=(0, 10.5), ylim=(-0.6, 7.5))
        xs = np.linspace(0, 10, 300)
        ax.plot(xs, np.maximum(0, 0.85 * (xs - 2.2)), color=C["blue"], lw=3)
        data_points(ax, x, y, s=110)
        ax.text(0.6, 6.2, "$\\hat y = \\max(0,\\ wx + b)$", fontsize=17, color=C["blue"])
        ax.annotate("flat at 0: no\nnegative prices", xy=(1.2, 0.05), xytext=(0.5, 2.6), fontsize=13,
                    color=C["grey"], arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.4))
        save(fig, "housing_relu")


INPUTS = ["size $x_1$", "bedrooms $x_2$", "postcode $x_3$", "wealth $x_4$"]


def fig_housing_networks():
    """1 neuron -> hidden layer with hand-made features -> fully connected hidden layer."""
    ys = [3.3, 2.1, 0.9, -0.3]
    hidden = [(5.0, 3.0), (5.0, 1.5), (5.0, 0.0)]
    feat = ["family size", "walkability", "school quality"]
    links = [(0, 0), (1, 0), (2, 1), (2, 2), (3, 2)]
    r = 0.42
    for step in (1, 2, 3):
        with sketchy():
            fig, ax = plt.subplots(figsize=(10, 4.6))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (-0.3, 10.6), (-1.4, 4.2))
            for (yk, lab) in zip(ys, INPUTS):
                box(ax, 1.3, yk, lab, 2.4, 0.7, fc="#E8F5E9", ec=C["green"], fs=14)
            out = (7.6, 1.5) if step > 1 else (5.6, 1.5)
            neuron(ax, *out, r, "out")
            arrow(ax, (out[0] + r, out[1]), (out[0] + 1.5, out[1]), lw=2)
            ax.text(out[0] + 1.6, out[1], "price $y$", va="center", fontsize=16)
            if step == 1:
                for yk in ys:
                    arrow(ax, (2.55, yk), tuple(np.array(out) - r * (np.array(out) - np.array((2.55, yk))) /
                                                 np.linalg.norm(np.array(out) - np.array((2.55, yk)))), lw=1.3, ms=11)
                ax.text(5.6, 2.4, "one neuron", ha="center", fontsize=14, color=C["red"])
            else:
                for (k, h) in enumerate(hidden):
                    neuron(ax, *h, r, "hidden")
                    connect(ax, h, out, r)
                pairs = links if step == 2 else [(i, j) for i in range(4) for j in range(3)]
                for i, j in pairs:
                    p0, h = np.array((2.55, ys[i])), np.array(hidden[j])
                    arrow(ax, tuple(p0), tuple(h - r * (h - p0) / np.linalg.norm(h - p0)), lw=1.2 if step == 3 else 1.5,
                          ms=11)
                if step == 2:
                    for (h, f) in zip(hidden, feat):
                        ax.text(h[0], h[1] + 0.55, f, ha="center", fontsize=13, color=C["blue"],
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=6)
                else:
                    ax.text(5.0, -1.05, "hidden layer: every input feeds every unit,\n"
                                        "the network learns its own features", ha="center", fontsize=13, color=C["blue"])
            save(fig, f"housing_network_step{step}")


# --------------------------------------------------------------------------- figures: representation
def _two_layer(ax, highlight=None, r=0.38, labels=True):
    xin = layer_positions(3, 1.0, 1.25, 1.5)
    hid = layer_positions(4, 4.0, 1.15, 1.5)
    out = (7.0, 1.5)
    for k, p in enumerate(xin):
        neuron(ax, *p, r, "in", f"$x_{k + 1}$", fs=15)
    for j, h in enumerate(hid):
        faded = highlight is not None and j != highlight
        neuron(ax, *h, r, "hidden", f"$a_{j + 1}^{{[1]}}$", fs=13, faded=faded)
        for k, p in enumerate(xin):
            on = highlight is None or j == highlight
            connect(ax, p, h, r, color="k" if on else FADE, lw=1.3 if on else 0.9)
            if highlight is not None and j == highlight:
                mid = 0.42 * np.array(p) + 0.58 * np.array(h)
                ax.text(mid[0], mid[1] + 0.14, f"$w_{{{j + 1}{k + 1}}}^{{[1]}}$", fontsize=12, color=C["purple"],
                        ha="center", bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.9), zorder=6)
        connect(ax, h, out, r, color="k" if highlight is None else FADE, lw=1.3 if highlight is None else 0.9)
    neuron(ax, *out, r, "out", "$a^{[2]}$", fs=13, faded=highlight is not None)
    arrow(ax, (out[0] + r, out[1]), (out[0] + 1.1, out[1]), color="k" if highlight is None else FADE, lw=1.8)
    ax.text(out[0] + 1.2, out[1], "$\\hat y$", va="center", fontsize=17, color="k" if highlight is None else FADE)
    if labels:
        for xc, t, a in [(1.0, "input layer", "$\\mathbf{a}^{[0]} = \\mathbf{x}$"), (4.0, "hidden layer", "$\\mathbf{a}^{[1]}$"),
                         (7.0, "output layer", "$a^{[2]} = \\hat y$")]:
            ax.text(xc, -1.25, t, ha="center", fontsize=14, color=C["orange"])
            ax.text(xc, -1.75, a, ha="center", fontsize=14)


def fig_two_layer():
    for step, hl in [(1, None), (2, 0), (3, 1)]:
        with sketchy():
            fig, ax = plt.subplots(figsize=(7.4, 5.0))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (0.2, 8.8), (-2.2, 4.1))
            _two_layer(ax, highlight=hl)
            save(fig, f"two_layer_step{step}")


def fig_neuron_zoom():
    with sketchy():
        fig, ax = plt.subplots(figsize=(8.6, 3.4))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 10.4), (-0.4, 3.6))
        r = 0.36
        xin = layer_positions(3, 0.8, 1.05, 1.6)
        for k, p in enumerate(xin):
            neuron(ax, *p, r, "in", f"$x_{k + 1}$", fs=14)
            arrow(ax, (p[0] + r, p[1]), (3.05, 1.6 + 0.25 * (1 - k)), lw=1.5)
        ax.add_patch(FancyBboxPatch((3.1, 0.75), 4.4, 1.7, boxstyle="round,pad=0.05,rounding_size=0.8",
                                    fc="#E3F2FD", ec=C["blue"], lw=2.2))
        ax.plot([5.3, 5.3], [0.85, 2.35], color=C["blue"], lw=2)
        ax.text(4.2, 1.6, "$z = \\mathbf{w}\\cdot\\mathbf{x} + b$", ha="center", va="center", fontsize=16)
        ax.text(6.4, 1.6, "$a = \\sigma(z)$", ha="center", va="center", fontsize=16)
        ax.text(4.2, 0.35, "weighted sum", ha="center", fontsize=13, color=C["grey"])
        ax.text(6.4, 0.35, "activation", ha="center", fontsize=13, color=C["grey"])
        arrow(ax, (7.55, 1.6), (8.8, 1.6), lw=2)
        ax.text(9.0, 1.6, "$a$", va="center", fontsize=18)
        ax.text(5.3, 3.05, "one neuron = logistic regression", ha="center", fontsize=15, color=C["red"])
        save(fig, "neuron_zoom")


def _deep(ax, sizes=(6, 7, 7, 7, 7, 7, 4), r=0.22, dx=1.3, dy=0.62):
    pos = []
    for l, n in enumerate(sizes):
        kind = "in" if l == 0 else ("out" if l == len(sizes) - 1 else "hidden")
        P = layer_positions(n, 0.6 + l * dx, dy, 2.2)
        for p in P:
            neuron(ax, *p, r, kind, lw=1.5)
        pos.append(P)
    for A, B in zip(pos[:-1], pos[1:]):
        for p in A:
            for q in B:
                d = np.array(q) - np.array(p)
                u = d / np.linalg.norm(d)
                ax.plot([p[0] + u[0] * r, q[0] - u[0] * r], [p[1] + u[1] * r, q[1] - u[1] * r], color=C["navy"],
                        lw=0.5, alpha=0.6, zorder=2)
    return pos


def fig_architectures():
    with sketchy():
        # 1. deep fully connected network
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.2, 9.4), (-0.6, 5.0))
        _deep(ax)
        ax.text(4.5, -0.35, "fully connected (dense) network: Lecture 7", ha="center", fontsize=15,
                color=C["navy"])
        save(fig, "architectures_step1")
        # 2. convolutional network
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.2, 9.4), (-0.6, 5.0))
        ax.add_patch(plt.Rectangle((0.2, 1.2), 1.6, 1.6, fc="#FFF8E1", ec="k", lw=2))
        ax.text(1.0, 2.0, "image", ha="center", va="center", fontsize=13)
        for k in range(4):
            ax.add_patch(plt.Rectangle((2.5 + 0.18 * k, 1.0 + 0.18 * k), 1.5, 1.5, fc="#E3F2FD", ec="k", lw=1.6))
        for k in range(6):
            ax.add_patch(plt.Rectangle((4.8 + 0.14 * k, 1.35 + 0.14 * k), 0.9, 0.9, fc="#E8F5E9", ec="k", lw=1.4))
        for k in range(8):
            neuron(ax, 7.3, 0.5 + 0.48 * k, 0.17, "hidden", lw=1.3)
        for k in range(3):
            neuron(ax, 8.6, 1.4 + 0.6 * k, 0.2, "out", lw=1.4)
        for xa, xb, y in [(1.85, 2.45, 2.0), (4.65, 4.75, 2.0), (6.55, 7.05, 2.2), (7.5, 8.35, 2.0)]:
            arrow(ax, (xa, y), (xb, y), lw=1.8)
        for xc, t in [(3.4, "convolution\n+ ReLU"), (5.6, "pooling"), (7.3, "dense"), (8.6, "classes")]:
            ax.text(xc, 0.1 if xc < 7 else -0.15, t, ha="center", fontsize=12)
        ax.text(4.5, -0.45 if False else 4.55, "convolutional neural network (CNN): Lecture 8", ha="center",
                fontsize=15, color=C["navy"])
        save(fig, "architectures_step2")
        # 3. recurrent network, unrolled in time
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.2, 9.4), (-0.6, 5.0))
        for k, t in enumerate(["t - 1", "t", "t + 1"]):
            xc = 2.2 + 2.6 * k
            neuron(ax, xc, 0.7, 0.36, "in", f"$x_{{{t}}}$", fs=12)
            box(ax, xc, 2.2, f"$h_{{{t}}}$", 1.0, 0.7, fc="#E3F2FD", ec=C["blue"], fs=13)
            neuron(ax, xc, 3.7, 0.36, "out", f"$o_{{{t}}}$", fs=12)
            arrow(ax, (xc, 1.08), (xc, 1.83), lw=1.8)
            arrow(ax, (xc, 2.57), (xc, 3.32), lw=1.8)
            if k < 2:
                arrow(ax, (xc + 0.55, 2.2), (xc + 2.05, 2.2), lw=1.8, color=C["blue"])
        arrow(ax, (0.6, 2.2), (1.65, 2.2), lw=1.8, color=C["blue"])
        arrow(ax, (7.85, 2.2), (8.9, 2.2), lw=1.8, color=C["blue"])
        ax.text(0.3, 2.2, "...", fontsize=18, va="center")
        ax.text(9.0, 2.2, "...", fontsize=18, va="center")
        ax.text(4.5, 4.55, "recurrent neural network (RNN), unrolled in time: Lecture 9", ha="center",
                fontsize=15, color=C["navy"])
        ax.text(4.5, -0.35, "the hidden state $h$ carries information along the sequence", ha="center",
                fontsize=13, color=C["blue"])
        save(fig, "architectures_step3")


# --------------------------------------------------------------------------- figures: data and activations
def fig_image_to_vector():
    from sklearn.datasets import load_digits

    img = load_digits().images[0].astype(int)          # an 8 x 8 "0" with grey values 0..16
    with sketchy():
        fig = plt.figure(figsize=(12, 4.6))
        ax = fig.add_axes([0.02, 0.06, 0.36, 0.78])
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=30)          # light shades, so dark digits stay readable
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(img[i, j]), ha="center", va="center", fontsize=11, color="k")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("8 x 8 image: 64 grey values", fontsize=15)
        ax2 = fig.add_axes([0.42, 0.42, 0.56, 0.2])
        flat = img.flatten()[:24]
        ax2.imshow(flat[None, :], cmap="gray_r", vmin=0, vmax=30, aspect="auto")
        for j, v in enumerate(flat):
            ax2.text(j, 0, str(v), ha="center", va="center", fontsize=11, color="k")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("flattened row by row: $\\mathbf{x} \\in \\mathbb{R}^{64}$ (first 24 entries)", fontsize=15)
        fig.text(0.40, 0.52, "$\\Rightarrow$", fontsize=28, ha="center", va="center")
        fig.text(0.7, 0.18, "MNIST: 28 x 28 images, so $\\mathbf{x} \\in \\mathbb{R}^{784}$", fontsize=15,
                 ha="center", color=C["navy"])
        save(fig, "image_to_vector")


def fig_activations():
    z = np.linspace(-5, 5, 400)
    funcs = [("sigmoid $\\sigma(z)$", 1 / (1 + np.exp(-z)), (-0.1, 1.1)),
             ("tanh$(z)$", np.tanh(z), (-1.15, 1.15)),
             ("ReLU$(z) = \\max(0, z)$", np.maximum(0, z), (-0.6, 5.2)),
             ("leaky ReLU, $\\alpha = 0.1$", np.where(z >= 0, z, 0.1 * z), (-0.8, 5.2))]
    with sketchy():
        fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.5))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.16, top=0.86, wspace=0.28)
        for ax, (title, f, yl) in zip(axs, funcs):
            ax.axhline(0, color=C["light"], lw=1)
            ax.axvline(0, color=C["light"], lw=1)
            ax.plot(z, f, color=C["red"], lw=3)
            ax.set_title(title, fontsize=15)
            ax.set_ylim(*yl)
            ax.set_xticks([-4, 0, 4])
            ax.set_xlabel("$z$")
            despine(ax)
        save(fig, "activations")


def fig_why_nonlinear():
    from sklearn.datasets import make_circles
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    X, y = make_circles(300, noise=0.12, factor=0.45, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    models = [("logistic regression", LogisticRegression()),
              ("8 hidden units, no activation", MLPClassifier((8,), activation="identity", max_iter=5000,
                                                               random_state=0)),
              ("8 hidden units, ReLU", MLPClassifier((8,), activation="relu", max_iter=5000, random_state=0))]
    lim = (-1.5, 1.5)
    g1, g2 = np.meshgrid(np.linspace(*lim, 300), np.linspace(*lim, 300))
    with sketchy():
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.04, top=0.88, wspace=0.08)
        for ax, (name, model) in zip(axs, models):
            model.fit(X_tr, y_tr)
            Z = model.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
            ax.contourf(g1, g2, Z, levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]], alpha=0.12)
            ax.contour(g1, g2, Z, levels=[0.5], colors="k", linewidths=2.2)
            ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], marker="o", s=30, facecolor="white",
                       edgecolor=C["blue"], linewidths=1.6, zorder=5)
            ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], marker="x", s=30, color=C["red"], linewidths=1.8,
                       zorder=5)
            ax.set_title(f"{name}: {model.score(X_te, y_te):.0%}", fontsize=15)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            despine(ax)
        save(fig, "why_nonlinear")


def _mnist_train():
    """The MNIST training images (60,000 x 28 x 28) and labels. Downloaded once into data/MNIST."""
    from torchvision import datasets

    ds = datasets.MNIST(root=ROOT / "data", train=True, download=True)
    return ds.data.numpy(), ds.targets.numpy()


def fig_mnist_digits():
    """Three examples of every digit from the MNIST training set."""
    X, y = _mnist_train()
    with sketchy():
        fig, axs = plt.subplots(3, 10, figsize=(10, 3.4))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.06, hspace=0.06)
        for d in range(10):
            idx = np.where(y == d)[0][:3]
            axs[0, d].set_title(str(d), fontsize=16)
            for r, i in enumerate(idx):
                axs[r, d].imshow(X[i], cmap="gray", vmin=0, vmax=255)
                axs[r, d].set_xticks([])
                axs[r, d].set_yticks([])
        save(fig, "mnist_digits")


def fig_zero_vs_one():
    """Binary problem: zeros (y = 0) versus ones (y = 1)."""
    X, y = _mnist_train()
    with sketchy():
        fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9))
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.02, top=0.98, wspace=0.06, hspace=0.08)
        for r, d in enumerate([0, 1]):
            for c, i in enumerate(np.where(y == d)[0][:4]):
                axs[r, c].imshow(X[i], cmap="gray", vmin=0, vmax=255)
                axs[r, c].set_xticks([])
                axs[r, c].set_yticks([])
            axs[r, 0].set_ylabel(f"$y = {d}$", fontsize=17, rotation=0, labelpad=30, va="center")
        save(fig, "zero_vs_one")


# =========================================================================== part 7b
# --------------------------------------------------------------------------- figures
def fig_simple_graph():
    """J = 3(a + bc) with a = 5, b = 3, c = 2."""
    leaves = {"a": (1.0, 3.6, "$a = 5$", "$\\partial J/\\partial a = 3$"),
              "b": (1.0, 2.2, "$b = 3$", "$\\partial J/\\partial b = 6$"),
              "c": (1.0, 0.8, "$c = 2$", "$\\partial J/\\partial c = 9$")}
    ops = {"u": (4.2, 1.5, "$u = bc$", "6", "$\\partial J/\\partial u = 3$"),
           "v": (7.0, 2.6, "$v = a + u$", "11", "$\\partial J/\\partial v = 3$"),
           "J": (9.8, 2.6, "$J = 3v$", "33", "")}
    W = {"a": 1.5, "b": 1.5, "c": 1.5, "u": 1.7, "v": 2.0, "J": 1.6}
    P = {k: v[:2] for k, v in {**leaves, **ops}.items()}
    edges = [("b", "u"), ("c", "u"), ("a", "v"), ("u", "v"), ("v", "J")]
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(11, 4.4))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (0, 11.2), (-0.2, 4.8))
            for p, q in edges:
                edge(ax, P[p], W[p], P[q], W[q])
            for k, (xc, yc, lab, grad) in leaves.items():
                node_box(ax, xc, yc, lab, W[k], kind="leaf")
                if step == 2:
                    red(ax, xc, yc - 0.4, grad)
            for k, (xc, yc, lab, val, grad) in ops.items():
                node_box(ax, xc, yc, lab, W[k], kind="loss" if k == "J" else "op")
                ax.text(xc, yc + 0.4, f"= {val}", ha="center", va="bottom", fontsize=14, color=C["navy"])
                if step == 2 and grad:
                    red(ax, xc, yc - 0.4, grad)
            ax.text(5.6, 4.45, "forward: values from left to right", ha="center", fontsize=14, color=C["navy"])
            if step == 2:
                ax.text(5.6, 0.05, "backward: derivatives of $J$ from right to left (chain rule)", ha="center",
                        fontsize=14, color=C["red"])
            save(fig, f"simple_graph_step{step}")


def fig_logreg_graph():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11.5, 3.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 12.2), (-0.4, 3.6))
        leaves = [(1.0, 2.6, "$\\mathbf{x}$", "data"), (1.0, 1.5, "$\\mathbf{w}$", "leaf"), (1.0, 0.4, "$b$", "leaf")]
        for xc, yc, t, k in leaves:
            node_box(ax, xc, yc, t, 0.9, kind=k)
        z = (4.4, 1.5, "$z = \\mathbf{w}\\cdot\\mathbf{x} + b$", 2.6)
        a = (7.6, 1.5, "$a = \\sigma(z)$", 1.9)
        L = (10.6, 1.5, "$L(a, y)$", 1.7)
        for xc, yc, _, _ in leaves:
            edge(ax, (xc, yc), 0.9, z[:2], z[3])
        edge(ax, z[:2], z[3], a[:2], a[3])
        edge(ax, a[:2], a[3], L[:2], L[3])
        node_box(ax, *z[:3], z[3])
        node_box(ax, *a[:3], a[3])
        node_box(ax, *L[:3], L[3], kind="loss")
        red(ax, 10.6, 1.05, "start", fs=12)
        red(ax, 7.6, 1.05, "$da = -\\frac{y}{a} + \\frac{1 - y}{1 - a}$")
        red(ax, 4.4, 1.05, "$dz = da\\,\\sigma'(z) = a - y$")
        red(ax, 1.75, 1.95, "$d\\mathbf{w} = (a - y)\\,\\mathbf{x}$", ha="left", va="center")
        red(ax, 1.75, 0.1, "$db = a - y$", ha="left", va="center")
        ax.text(6.1, 3.25, "notation: $d\\theta = \\partial L / \\partial \\theta$ for one example", ha="center",
                fontsize=14, color=C["red"])
        save(fig, "logreg_graph")


def fig_backprop_flow():
    """Two-layer network as a chain; backward arrows reuse every intermediate derivative."""
    chain = [("$\\mathbf{a}^{[0]}$", "data", 0.9), ("$\\mathbf{z}^{[1]}$", "op", 0.9), ("$\\mathbf{a}^{[1]}$", "op", 0.9),
             ("$\\mathbf{z}^{[2]}$", "op", 0.9), ("$a^{[2]}$", "op", 0.9), ("$L$", "loss", 0.8)]
    xs = [0.8, 3.0, 5.2, 7.4, 9.6, 11.6]
    y0 = 1.6
    with sketchy():
        fig, ax = plt.subplots(figsize=(12, 4.4))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 12.4), (-0.6, 4.2))
        for (t, k, w), x in zip(chain, xs):
            node_box(ax, x, y0, t, w, kind=k, fs=16)
        for i in range(5):
            edge(ax, (xs[i], y0), chain[i][2], (xs[i + 1], y0), chain[i + 1][2])
        for x, lab in [(3.0, 1), (7.4, 2)]:
            node_box(ax, x - 0.55, 3.3, f"$\\mathbf{{W}}^{{[{lab}]}}$", 0.9, kind="leaf", fs=14)
            node_box(ax, x + 0.55, 3.3, f"$\\mathbf{{b}}^{{[{lab}]}}$", 0.8, kind="leaf", fs=14)
            arrow(ax, (x - 0.45, 2.95), (x - 0.12, y0 + 0.36), lw=1.5, ms=13)
            arrow(ax, (x + 0.45, 2.95), (x + 0.12, y0 + 0.36), lw=1.5, ms=13)
            red(ax, x - 1.15, 3.3, f"$d\\mathbf{{W}}^{{[{lab}]}}$", ha="right", va="center")
            red(ax, x + 1.1, 3.3, f"$d\\mathbf{{b}}^{{[{lab}]}}$", ha="left", va="center")
        for i in range(5, 1, -1):
            arrow(ax, (xs[i] - 0.25, y0 - 0.4), (xs[i - 1] + 0.25, y0 - 0.4), color=C["red"], lw=2, ms=15, rad=-0.35)
        arrow(ax, (xs[1] - 0.25, y0 - 0.4), (xs[1] - 0.4, y0 - 0.35), color=C["red"], lw=0.01, ms=1)
        for x, lab in zip(xs[1:5], ["$d\\mathbf{z}^{[1]}$", "$d\\mathbf{a}^{[1]}$", "$d\\mathbf{z}^{[2]}$", "$da^{[2]}$"]):
            red(ax, x, y0 - 0.95, lab, fs=14)
        ax.text(6.2, -0.45, "backward pass: every $d(\\cdot)$ is computed once and reused", ha="center",
                fontsize=14, color=C["red"])
        save(fig, "backprop_flow")


def fig_deep_network():
    sizes = [3, 5, 5, 3, 1]
    r, dx, dy = 0.28, 2.1, 0.78
    with sketchy():
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.2, 10.6), (-1.6, 4.4))
        pos = []
        for l, n in enumerate(sizes):
            kind = "in" if l == 0 else ("out" if l == len(sizes) - 1 else "hidden")
            P = layer_positions(n, 0.8 + l * dx, dy, 1.8)
            for p in P:
                neuron(ax, *p, r, kind, lw=1.8)
            pos.append(P)
            ax.text(0.8 + l * dx, -0.85, f"$l = {l}$", ha="center", fontsize=14, color=C["orange"])
            ax.text(0.8 + l * dx, -1.35, f"$n^{{[{l}]}} = {n}$", ha="center", fontsize=14)
        for A, B in zip(pos[:-1], pos[1:]):
            for p in A:
                for q in B:
                    connect(ax, p, q, r, lw=0.8, ms=7)
        arrow(ax, (pos[-1][0][0] + r, pos[-1][0][1]), (pos[-1][0][0] + 1.0, pos[-1][0][1]), lw=1.8)
        ax.text(pos[-1][0][0] + 1.1, pos[-1][0][1], "$\\hat y$", fontsize=17, va="center")
        ax.text(5.0, 4.15, "$L = 4$ layers (the input layer is not counted)", ha="center", fontsize=14,
                color=C["navy"])
        save(fig, "deep_network")


def fig_blocks():
    """Forward and backward pass as a chain of layer blocks."""
    with sketchy():
        fig, ax = plt.subplots(figsize=(12, 4.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 12.4), (-0.4, 4.6))
        xs = [2.4, 5.4, 9.0]
        names = ["layer 1", "layer $l$", "layer $L$"]
        for x, n in zip(xs, names):
            box(ax, x, 3.3, n, 1.6, 0.75, fc="#E3F2FD", ec=C["blue"], fs=14)
            box(ax, x, 1.1, n, 1.6, 0.75, fc="#FCE4EC", ec=C["red"], fs=14)
            arrow(ax, (x, 2.9), (x, 1.5), color=C["grey"], lw=1.4, ms=12)
            ax.text(x + 0.1, 2.2, "cache", fontsize=11, color=C["grey"], va="center")
        ax.text(0.55, 3.3, "$\\mathbf{A}^{[0]}$", fontsize=15, va="center")
        arrow(ax, (1.0, 3.3), (1.57, 3.3), lw=1.8)
        ax.text(3.9, 3.3, "...", fontsize=18, ha="center", va="center")
        arrow(ax, (3.23, 3.3), (3.7, 3.3), lw=1.8)
        arrow(ax, (4.1, 3.3), (4.57, 3.3), lw=1.8)
        ax.text(6.6, 3.62, "$\\mathbf{A}^{[l]}$", fontsize=14, ha="center")
        arrow(ax, (6.23, 3.3), (7.0, 3.3), lw=1.8)
        ax.text(7.4, 3.3, "...", fontsize=18, ha="center", va="center")
        arrow(ax, (7.75, 3.3), (8.17, 3.3), lw=1.8)
        arrow(ax, (9.83, 3.3), (10.5, 3.3), lw=1.8)
        box(ax, 11.3, 3.3, "$J$", 1.0, 0.7, fc="#FCE4EC", ec=C["red"], fs=15)
        ax.text(4.5, 4.15, "forward: $\\mathbf{Z}^{[l]} = \\mathbf{A}^{[l-1]}(\\mathbf{W}^{[l]})^{\\top} + \\mathbf{b}^{[l]}$, "
                           "$\\mathbf{A}^{[l]} = g^{[l]}(\\mathbf{Z}^{[l]})$", ha="center", fontsize=14, color=C["navy"])
        arrow(ax, (11.3, 2.92), (10.3, 1.35), color=C["red"], lw=1.8, rad=-0.2)
        ax.text(10.95, 1.95, "$d\\mathbf{A}^{[L]}$", fontsize=13, color=C["red"])
        for (xa, xb) in [(8.17, 7.75), (7.0, 6.23), (4.57, 4.1), (3.7, 3.23)]:
            arrow(ax, (xa, 1.1), (xb, 1.1), color=C["red"], lw=1.8)
        ax.text(7.4, 1.1, "...", fontsize=18, ha="center", va="center", color=C["red"])
        ax.text(3.9, 1.1, "...", fontsize=18, ha="center", va="center", color=C["red"])
        ax.text(6.6, 1.38, "$d\\mathbf{A}^{[l-1]}$", fontsize=13, ha="center", color=C["red"])
        for x, l in zip(xs, ["1", "l", "L"]):
            ax.text(x, 0.45, f"$d\\mathbf{{W}}^{{[{l}]}},\\ d\\mathbf{{b}}^{{[{l}]}}$", ha="center", va="top",
                    fontsize=13, color=C["red"])
        ax.text(6.2, -0.25, "backward: each layer receives $d\\mathbf{A}^{[l]}$, returns $d\\mathbf{A}^{[l-1]}$ and its own "
                           "gradients", ha="center", fontsize=14, color=C["red"])
        save(fig, "blocks")


def fig_vanishing():
    import torch
    from torch import nn

    torch.manual_seed(0)
    Xb = torch.randn(256, 64)
    yb = torch.randint(0, 2, (256, 1)).float()
    results = []
    for label, act, he in [("sigmoid", nn.Sigmoid, False), ("ReLU, default init", nn.ReLU, False),
                           ("ReLU, He init", nn.ReLU, True)]:
        torch.manual_seed(0)
        layers = []
        for _ in range(10):
            lin = nn.Linear(64, 64)
            if he:
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
            layers += [lin, act()]
        net = nn.Sequential(*layers, nn.Linear(64, 1))
        nn.functional.binary_cross_entropy_with_logits(net(Xb), yb).backward()
        results.append((label, [net[2 * l].weight.grad.norm().item() for l in range(10)]))
    cols = [C["red"], C["orange"], C["green"]]
    with sketchy():
        fig, ax = plt.subplots(figsize=(8, 4.4))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        for (label, g), col in zip(results, cols):
            ax.semilogy(np.arange(1, 11), g, color=col, lw=2.8, marker="o", ms=6, label=label)
        ax.set_xlabel("layer $l$ (1 = closest to the input)")
        ax.set_ylabel("$\\|\\partial J / \\partial \\mathbf{W}^{[l]}\\|$ (log scale)")
        ax.set_xticks(range(1, 11))
        ax.legend(loc="lower right", fontsize=13, frameon=False)
        despine(ax)
        save(fig, "vanishing")
    for label, g in results:
        print(f"    {label}: layer 1 {g[0]:.1e}, layer 10 {g[-1]:.1e}")


def _train_digits(epochs=30):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    y = torch.tensor(digits.target)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=digits.target)
    loader = DataLoader(TensorDataset(X[idx_tr], y[idx_tr]), batch_size=64, shuffle=True)
    X_te, y_te = X[idx_te], y[idx_te]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hist = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for Xb, yb in loader:
            loss = loss_fn(model(Xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            pred_te = model(X_te).argmax(dim=1)
            acc = (pred_te == y_te).float().mean().item()
            tr_acc = (model(X[idx_tr]).argmax(dim=1) == y[idx_tr]).float().mean().item()
        hist.append((total / len(idx_tr), tr_acc, acc))
    return np.array(hist), digits.images[idx_te], y_te.numpy(), pred_te.numpy()


def fig_digits_training():
    hist, imgs, y_te, pred = _train_digits()
    ep = np.arange(1, len(hist) + 1)
    with sketchy():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.9, wspace=0.25)
        a1.plot(ep, hist[:, 0], color=C["blue"], lw=3)
        a1.set_xlabel("epoch")
        a1.set_ylabel("training loss")
        a1.set_title("cross-entropy loss", fontsize=16)
        despine(a1)
        a2.plot(ep, 100 * hist[:, 1], color=C["blue"], lw=2.6, label="training set")
        a2.plot(ep, 100 * hist[:, 2], color=C["red"], lw=2.6, label="test set")
        a2.set_ylim(0, 102)
        a2.set_xlabel("epoch")
        a2.set_ylabel("accuracy (%)")
        a2.set_title(f"test accuracy after {len(hist)} epochs: {100 * hist[-1, 2]:.1f}%", fontsize=16)
        a2.legend(loc="lower right", fontsize=13, frameon=False)
        despine(a2)
        save(fig, "digits_training")
        fig, axs = plt.subplots(2, 8, figsize=(11, 3.4))
        fig.subplots_adjust(0.01, 0.02, 0.99, 0.9, wspace=0.15, hspace=0.45)
        wrong = np.where(pred != y_te)[0][:4]
        right = np.where(pred == y_te)[0][:16 - len(wrong)]
        show = np.r_[right[:12], wrong]
        for ax, i in zip(axs.ravel(), show):
            ax.imshow(imgs[i], cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            ok = pred[i] == y_te[i]
            ax.set_title(f"pred {pred[i]}" + ("" if ok else f" (is {y_te[i]})"), fontsize=12,
                         color=C["green"] if ok else C["red"])
        save(fig, "digits_predictions")
    print(f"    final: train loss {hist[-1, 0]:.4f}, train acc {hist[-1, 1]:.4f}, test acc {hist[-1, 2]:.4f}; "
          f"wrong {int((pred != y_te).sum())} of {len(y_te)}")


# =========================================================================== part 7c (and appendix)
# --------------------------------------------------------------------------- long version
def fig_chain():
    """Two-layer network as a chain of mappings: local ('easy') derivatives in blue, accumulated ones in red."""
    xs = [0.8, 3.1, 5.4, 7.7, 10.0, 12.0]
    names = ["$\\mathbf{a}^{[0]}$", "$\\mathbf{z}^{[1]}$", "$\\mathbf{a}^{[1]}$", "$\\mathbf{z}^{[2]}$", "$a^{[2]}$", "$L$"]
    kinds = ["data", "op", "op", "op", "op", "loss"]
    local = [None, "$g'(\\mathbf{z}^{[1]})$", "$\\mathbf{W}^{[2]}$", "$\\sigma'(z^{[2]})$", "$\\partial L/\\partial a^{[2]}$"]
    acc = {4: "$\\partial L/\\partial a^{[2]}$", 3: "$\\partial L/\\partial z^{[2]}$", 2: "$\\partial L/\\partial \\mathbf{a}^{[1]}$",
           1: "$\\partial L/\\partial \\mathbf{z}^{[1]}$"}
    y0 = 1.7
    for step in (1, 2, 3):
        with sketchy():
            fig, ax = plt.subplots(figsize=(12.5, 4.6))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (0, 12.9), (-0.7, 4.4))
            for x, t, k in zip(xs, names, kinds):
                node_box(ax, x, y0, t, 0.95, kind=k, fs=16)
            for i in range(5):
                edge(ax, (xs[i], y0), 0.95, (xs[i + 1], y0), 0.95)
                if local[i]:
                    ax.text((xs[i] + xs[i + 1]) / 2, y0 + 0.12, local[i], ha="center", va="bottom", fontsize=12,
                            color=C["blue"])
            for x, lab in [(3.1, 1), (7.7, 2)]:
                node_box(ax, x - 0.6, 3.5, f"$\\mathbf{{W}}^{{[{lab}]}}$", 0.95, kind="leaf", fs=14)
                node_box(ax, x + 0.6, 3.5, f"$\\mathbf{{b}}^{{[{lab}]}}$", 0.85, kind="leaf", fs=14)
                arrow(ax, (x - 0.5, 3.15), (x - 0.12, y0 + 0.36), lw=1.5, ms=13)
                arrow(ax, (x + 0.5, 3.15), (x + 0.12, y0 + 0.36), lw=1.5, ms=13)
            show = {1: [], 2: [4, 3], 3: [4, 3, 2, 1]}[step]
            for i in show:
                red(ax, xs[i], y0 - 0.45, acc[i], fs=13)
            if step >= 2:
                for i in range(5, 3, -1):
                    arrow(ax, (xs[i] - 0.3, y0 - 0.95), (xs[i - 1] + 0.3, y0 - 0.95), color=C["red"], lw=2, ms=14)
                red(ax, 7.7 - 1.25, 3.5, "$\\partial L/\\partial \\mathbf{W}^{[2]}$", ha="right", va="center")
                red(ax, 7.7 + 1.15, 3.5, "$\\partial L/\\partial \\mathbf{b}^{[2]}$", ha="left", va="center")
            if step == 3:
                for i in range(3, 1, -1):
                    arrow(ax, (xs[i] - 0.3, y0 - 0.95), (xs[i - 1] + 0.3, y0 - 0.95), color=C["red"], lw=2, ms=14)
                red(ax, 3.1 - 1.25, 3.5, "$\\partial L/\\partial \\mathbf{W}^{[1]}$", ha="right", va="center")
                red(ax, 3.1 + 1.15, 3.5, "$\\partial L/\\partial \\mathbf{b}^{[1]}$", ha="left", va="center")
            caption = {1: "forward: a chain of simple mappings (blue: the local, 'easy' derivative of each step)",
                       2: "layer 2: two steps back from the loss",
                       3: "layer 1: the same steps, continued one layer further"}[step]
            ax.text(6.4, -0.55, caption, ha="center", fontsize=14, color=C["navy"] if step == 1 else C["red"])
            save(fig, f"chain_step{step}")


def fig_chain_deep():
    """L = 4: every layer repeats the same backward step."""
    with sketchy():
        fig, ax = plt.subplots(figsize=(12.5, 3.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 12.9), (-0.6, 3.4))
        xs = [0.6, 2.9, 5.2, 7.5, 9.8, 12.2]
        labels = ["$\\mathbf{a}^{[0]}$", "$\\mathbf{a}^{[1]}$", "$\\mathbf{a}^{[2]}$", "$\\mathbf{a}^{[3]}$", "$\\mathbf{a}^{[4]}$", "$L$"]
        for x, t in zip(xs, labels):
            node_box(ax, x, 1.4, t, 0.9, kind="loss" if t == "$L$" else ("data" if x == 0.6 else "op"), fs=15)
        for l in range(4):
            xm = (xs[l] + xs[l + 1]) / 2
            box(ax, xm, 2.75, f"layer {l + 1}: $\\mathbf{{W}}^{{[{l + 1}]}}, \\mathbf{{b}}^{{[{l + 1}]}}$", 2.0, 0.55,
                fc="#E3F2FD", ec=C["blue"], fs=12)
            arrow(ax, (xm, 2.45), (xm, 1.55), lw=1.4, ms=12)
            red(ax, xm, 0.95, f"$d\\mathbf{{W}}^{{[{l + 1}]}}, d\\mathbf{{b}}^{{[{l + 1}]}}$", fs=12)
        for i in range(5):
            edge(ax, (xs[i], 1.4), 0.9, (xs[i + 1], 1.4), 0.9)
        for i in range(5, 0, -1):
            arrow(ax, (xs[i] - 0.3, 0.35), (xs[i - 1] + 0.3, 0.35), color=C["red"], lw=2, ms=14)
        ax.text(6.4, -0.4, "backward: $d\\mathbf{a}^{[l]} \\to d\\mathbf{z}^{[l]} \\to d\\mathbf{W}^{[l]}, d\\mathbf{b}^{[l]}$ "
                           "and $d\\mathbf{a}^{[l-1]}$, for $l = 4, 3, 2, 1$", ha="center", fontsize=14, color=C["red"])
        save(fig, "chain_deep")


# --------------------------------------------------------------------------- appendix
def _matrix_panel(ax, M, title, vmax=None):
    v = vmax if vmax is not None else 2.2 * np.abs(M).max()
    ax.imshow(np.abs(M), cmap="Blues", vmin=0, vmax=v)
    for (i, j), val in np.ndenumerate(M):
        ax.text(j, i, "0" if abs(val) < 1e-12 else f"{val:.2f}", ha="center", va="center", fontsize=12, color="k")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15)


def fig_jacobians():
    rng = np.random.default_rng(3)
    W = np.round(rng.normal(0, 0.6, size=(4, 3)), 2)
    z = rng.normal(size=4)
    s = 1 / (1 + np.exp(-z))
    D = np.diag(s * (1 - s))
    p = np.exp([2.0, 1.0, 0.1])
    p = p / p.sum()
    S = np.diag(p) - np.outer(p, p)
    with sketchy():
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.0), gridspec_kw=dict(width_ratios=[3, 4, 3]))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.84, wspace=0.25)
        _matrix_panel(axs[0], W, "$\\partial\\mathbf{z}/\\partial\\mathbf{a} = \\mathbf{W}$ (dense)")
        _matrix_panel(axs[1], D, "$\\partial\\mathbf{a}/\\partial\\mathbf{z} = \\mathrm{diag}(\\sigma'(\\mathbf{z}))$")
        _matrix_panel(axs[2], S, "softmax: $\\mathrm{diag}(\\mathbf{p}) - \\mathbf{p}\\mathbf{p}^{\\top}$")
        save(fig, "jacobians")


def fig_gradcheck():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5)
    yv = np.array([0, 0, 1.0])
    A = rng.normal(size=(8, 5)) * 0.5
    c = rng.normal(size=8) * 0.1
    B = rng.normal(size=(3, 8)) * 0.5
    e = rng.normal(size=3) * 0.1

    def loss(A):
        h = np.tanh(A @ x + c)
        s = B @ h + e
        s = s - s.max()
        return -np.log((np.exp(s) / np.exp(s).sum()) @ yv)

    h = np.tanh(A @ x + c)
    s = B @ h + e
    pr = np.exp(s - s.max())
    pr /= pr.sum()
    gA = np.outer((B.T @ (pr - yv)) * (1 - h ** 2), x)          # backpropagation
    eps = np.logspace(-12, -1, 23)
    central, forward = [], []
    for ep in eps:
        nc, nf = np.zeros_like(A), np.zeros_like(A)
        L0 = loss(A)
        for idx in np.ndindex(A.shape):
            old = A[idx]
            A[idx] = old + ep
            Lp = loss(A)
            A[idx] = old - ep
            Lm = loss(A)
            A[idx] = old
            nc[idx] = (Lp - Lm) / (2 * ep)
            nf[idx] = (Lp - L0) / ep
        rel = lambda n: np.linalg.norm(n - gA) / (np.linalg.norm(n) + np.linalg.norm(gA))  # noqa: E731
        central.append(rel(nc))
        forward.append(rel(nf))
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.95)
        ax.loglog(eps, forward, color=C["orange"], lw=2.6, marker="o", ms=4, label="forward difference")
        ax.loglog(eps, central, color=C["blue"], lw=2.6, marker="o", ms=4, label="central difference")
        ax.set_xlabel("step size $\\varepsilon$")
        ax.set_ylabel("relative error vs. backprop")
        ax.legend(loc="upper center", fontsize=13, frameon=False)
        ax.text(1e-11, 1e-10, "rounding errors\ndominate", fontsize=12, color=C["grey"])
        ax.text(3e-4, 1e-10, "truncation errors\ndominate", fontsize=12, color=C["grey"])
        despine(ax)
        save(fig, "gradcheck")
    i = int(np.argmin(central))
    print(f"    best central: {central[i]:.1e} at eps {eps[i]:.0e}; best forward: {min(forward):.1e}")


FIGURES = {   # name: (output folder, function)
    "housing_relu": (OUT_A, fig_housing_relu),
    "housing_network": (OUT_A, fig_housing_networks),
    "two_layer": (OUT_A, fig_two_layer),
    "neuron_zoom": (OUT_A, fig_neuron_zoom),
    "architectures": (OUT_A, fig_architectures),
    "image_to_vector": (OUT_A, fig_image_to_vector),
    "activations": (OUT_A, fig_activations),
    "why_nonlinear": (OUT_A, fig_why_nonlinear),
    "mnist_digits": (OUT_A, fig_mnist_digits),
    "zero_vs_one": (OUT_A, fig_zero_vs_one),
    "simple_graph": (OUT_B, fig_simple_graph),
    "logreg_graph": (OUT_B, fig_logreg_graph),
    "backprop_flow": (OUT_B, fig_backprop_flow),
    "deep_network": (OUT_B, fig_deep_network),
    "blocks": (OUT_B, fig_blocks),
    "vanishing": (OUT_B, fig_vanishing),
    "digits": (OUT_B, fig_digits_training),
    "chain_step": (OUT_C, fig_chain),
    "chain_deep": (OUT_C, fig_chain_deep),
    "jacobians": (OUT_C, fig_jacobians),
    "gradcheck": (OUT_C, fig_gradcheck),
}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}")
    for key, (folder, fn) in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            OUT = folder                    # save() writes to the folder of this figure
            print(f"{key}  ->  {folder.relative_to(ROOT)}")
            fn()
