"""
Sketch-style figures for Lecture 9: Recurrent neural networks.
    part 9a   sequence data, the RNN, forward propagation, backpropagation through time, RNN types
    part 9b   language models, sampling, a character-level RNN that invents dinosaur names
    part 9c   vanishing gradients, GRU, LSTM, PyTorch cells, bidirectional and deep RNNs

Run from anywhere:
    python scripts/make_figures_lecture09.py                  # all figures (the experiments take a few minutes)
    python scripts/make_figures_lecture09.py unrolled gru     # only figures whose name contains these words

Output: figures/rnn-sketch/*.png (relative to the project root).

Requirements: numpy, matplotlib >= 3.7; PyTorch for the three experiments (dinosaur names and the two
gradient-norm figures). Fixed seeds, so the numbers quoted on the slides are reproducible.
Data: data/dinosaurs.txt, about 300 dinosaur genus names (one per line).

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
OUT = ROOT / "figures" / "rnn-sketch"
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
DATA = ROOT / "data" / "dinosaurs.txt"


def box(ax, xc, yc, text, w, h, fc="#E3F2FD", ec=None, fs=14, lw=2, color="k", rounding=0.12):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle=f"round,pad=0.03,rounding_size={rounding}",
                                fc=fc, ec=ec or C["blue"], lw=lw, zorder=3))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(ax, p, q, color="k", lw=1.8, ms=14, rad=0.0, ls="-"):
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=ms, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def circle(ax, x, y, text, r=0.38, fc="#E8F5E9", ec=None, fs=13):
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec or C["green"], lw=1.8, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def _unrolled(ax, T, xs=None, ys=None, params=False, x0=1.6, dx=2.3, y_in=0.5, y_cell=2.0, y_out=3.5,
              in_labels=None, out_labels=None, a0=True, loss=False, back=False, out_fc="#FCE4EC"):
    """An RNN unrolled over T time steps. Returns the x positions of the cells."""
    pos = [x0 + dx * t for t in range(T)]
    if a0:
        ax.text(x0 - 1.55, y_cell, "$\\mathbf{a}^{\\langle 0 \\rangle}$", ha="center", va="center", fontsize=15)
        arrow(ax, (x0 - 1.15, y_cell), (x0 - 0.52, y_cell))
    for t, xc in enumerate(pos, start=1):
        box(ax, xc, y_cell, f"$\\mathbf{{a}}^{{\\langle {t} \\rangle}}$", 1.0, 0.8, fs=14)
        if xs is not False:
            lab = in_labels[t - 1] if in_labels else f"$\\mathbf{{x}}^{{\\langle {t} \\rangle}}$"
            circle(ax, xc, y_in, lab, fs=12 if in_labels else 13)
            arrow(ax, (xc, y_in + 0.38), (xc, y_cell - 0.42))
        if ys is not False:
            lab = out_labels[t - 1] if out_labels else f"$\\hat{{\\mathbf{{y}}}}^{{\\langle {t} \\rangle}}$"
            circle(ax, xc, y_out, lab, fc=out_fc, ec=C["red"], fs=12 if out_labels else 13)
            arrow(ax, (xc, y_cell + 0.42), (xc, y_out - 0.38))
        if t < T:
            arrow(ax, (xc + 0.52, y_cell), (xc + dx - 0.52, y_cell))
        if params:
            ax.text(xc + 0.12, (y_in + y_cell) / 2 - 0.05, "$\\mathbf{W}_{ax}$", fontsize=11, color=C["purple"])
            ax.text(xc + 0.12, (y_cell + y_out) / 2 - 0.05, "$\\mathbf{W}_{ya}$", fontsize=11, color=C["purple"])
            if t < T:
                ax.text(xc + dx / 2, y_cell + 0.12, "$\\mathbf{W}_{aa}$", fontsize=11, color=C["purple"], ha="center")
        if loss:
            box(ax, xc, y_out + 1.25, f"$L^{{\\langle {t} \\rangle}}$", 0.9, 0.6, fc="#FFF8E1", ec=C["orange"], fs=12)
            arrow(ax, (xc, y_out + 0.38), (xc, y_out + 0.93))
        if back and t > 1:
            arrow(ax, (xc - 0.4, y_cell - 0.55), (xc - dx + 0.4, y_cell - 0.55), color=C["red"], lw=2.2, rad=0.25)
    return pos


# --------------------------------------------------------------------------- 10a
def fig_one_hot():
    vocab = [("a", 1), ("aaron", 2), ("...", None), ("and", 367), ("...", None), ("granger", 3681),
             ("...", None), ("harry", 4075), ("...", None), ("potter", 6830), ("...", None), ("zulu", 10000)]
    with sketchy():
        fig, ax = plt.subplots(figsize=(11.5, 4.4))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 15), (-0.4, 6.0))
        ax.text(1.5, 5.6, "vocabulary", ha="center", fontsize=15, color=C["navy"])
        for k, (w, idx) in enumerate(vocab):
            y = 5.0 - 0.43 * k
            ax.text(0.4, y, "" if idx is None else str(idx), ha="right", fontsize=11, color=C["grey"])
            ax.text(0.6, y, w, fontsize=12, va="center", color=C["red"] if w in ("harry", "potter") else "k")
        for j, (word, hit) in enumerate([("harry", 7), ("potter", 9)]):
            x0 = 6.0 + 4.0 * j
            ax.text(x0 + 0.35, 5.6, f"one-hot vector of \"{word}\"", ha="center", fontsize=13, color=C["navy"])
            for k in range(len(vocab)):
                y = 5.0 - 0.43 * k
                v = "..." if vocab[k][1] is None else ("1" if k == hit else "0")
                ax.add_patch(plt.Rectangle((x0, y - 0.2), 0.7, 0.4, fc="#FCE4EC" if v == "1" else "white", ec="k",
                                           lw=1.0))
                ax.text(x0 + 0.35, y, v, ha="center", va="center", fontsize=11)
        ax.text(12.4, 2.6, "$\\mathbf{x}^{\\langle t \\rangle} \\in \\{0, 1\\}^{10\\,000}$\nexactly one entry is 1",
                fontsize=13, va="center")
        save(fig, "one_hot")


def fig_dense_on_sequence():
    with sketchy():
        fig, ax = plt.subplots(figsize=(10.5, 4.0))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 12), (-0.6, 4.4))
        for k, lab in enumerate(["$\\mathbf{x}^{\\langle 1 \\rangle}$", "$\\mathbf{x}^{\\langle 2 \\rangle}$", "...",
                                 "$\\mathbf{x}^{\\langle T_x \\rangle}$"]):
            y = 3.5 - 1.0 * k
            if lab == "...":
                ax.text(1.0, y, "...", ha="center", va="center", fontsize=16)
            else:
                circle(ax, 1.0, y, lab)
            arrow(ax, (1.45, y), (3.3, 2.0), lw=1.2)
        box(ax, 5.0, 2.0, "standard\nneural network", 3.2, 2.2, fs=15)
        for k, lab in enumerate(["$\\hat y^{\\langle 1 \\rangle}$", "$\\hat y^{\\langle 2 \\rangle}$", "...",
                                 "$\\hat y^{\\langle T_y \\rangle}$"]):
            y = 3.5 - 1.0 * k
            arrow(ax, (6.65, 2.0), (8.5, y), lw=1.2)
            if lab == "...":
                ax.text(9.0, y, "...", ha="center", va="center", fontsize=16)
            else:
                circle(ax, 9.0, y, lab, fc="#FCE4EC", ec=C["red"])
        ax.text(10.2, 2.0, "all words at\nonce, as one\nlong vector", fontsize=12, va="center", color=C["grey"])
        save(fig, "dense_on_sequence")


def fig_unrolled():
    for step in (1, 2):
        with sketchy():
            fig, ax = plt.subplots(figsize=(11.5, 4.3))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (-0.4, 12.0), (-0.1, 4.2))
            _unrolled(ax, 4, params=step == 2, x0=1.6, dx=2.9)
            ax.text(11.6, 2.0, "...", fontsize=18, va="center", ha="center")
            if step == 2:
                ax.text(5.9, -0.0, "the same parameters $\\mathbf{W}_{aa}, \\mathbf{W}_{ax}, \\mathbf{W}_{ya}$ "
                                   "(and biases) at every time step", ha="center", fontsize=13, color=C["purple"])
            save(fig, f"unrolled_step{step}")


def fig_rnn_cell():
    with sketchy():
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 12.5), (-0.3, 5.4))
        ax.add_patch(FancyBboxPatch((2.4, 0.9), 6.8, 3.7, boxstyle="round,pad=0.05,rounding_size=0.3", fc="#F5F9FF",
                                    ec=C["blue"], lw=2, ls="--"))
        ax.text(0.9, 2.6, "$\\mathbf{a}^{\\langle t-1 \\rangle}$", fontsize=16, ha="center", va="center")
        arrow(ax, (1.4, 2.6), (3.1, 2.6))
        circle(ax, 5.0, -0.0 + 0.3, "$\\mathbf{x}^{\\langle t \\rangle}$", r=0.42)
        box(ax, 3.9, 2.6, "$\\mathbf{W}_{aa}\\,\\cdot$", 1.3, 0.7, fc="white", fs=13)
        box(ax, 5.0, 1.45, "$\\mathbf{W}_{ax}\\,\\cdot$", 1.3, 0.6, fc="white", fs=13)
        arrow(ax, (5.0, 0.72), (5.0, 1.12))
        circle(ax, 5.6, 2.6, "$+$", r=0.3, fc="white", ec="k", fs=15)
        arrow(ax, (4.58, 2.6), (5.27, 2.6))
        arrow(ax, (5.2, 1.78), (5.45, 2.33))
        ax.text(5.6, 3.25, "$+\\,\\mathbf{b}_a$", fontsize=12, ha="center")
        box(ax, 7.0, 2.6, "tanh", 1.1, 0.7, fc="#FFF8E1", ec=C["orange"], fs=14)
        arrow(ax, (5.92, 2.6), (6.42, 2.6))
        arrow(ax, (7.58, 2.6), (10.3, 2.6))
        ax.text(11.0, 2.6, "$\\mathbf{a}^{\\langle t \\rangle}$", fontsize=16, ha="center", va="center")
        box(ax, 8.4, 3.9, "$\\mathbf{W}_{ya}\\cdot + \\mathbf{b}_y$, softmax", 2.8, 0.7, fc="#FCE4EC", ec=C["red"], fs=12)
        arrow(ax, (8.4, 2.6), (8.4, 3.53))
        arrow(ax, (9.82, 3.9), (10.4, 4.6))
        ax.text(11.0, 4.75, "$\\hat{\\mathbf{y}}^{\\langle t \\rangle}$", fontsize=16, ha="center")
        save(fig, "rnn_cell")


def fig_bptt():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.4, 12.4), (-0.2, 6.4))
        pos = _unrolled(ax, 4, loss=True, back=True, x0=1.6, dx=2.9)
        box(ax, 5.95, 5.95, "$L = \\sum_t L^{\\langle t \\rangle}$", 2.2, 0.6, fc="#FFF8E1", ec=C["orange"], fs=13)
        for xc in pos:
            arrow(ax, (xc, 5.05), (5.95 + 0.25 * (xc - 5.95) / 4.35, 5.63), lw=1.0, color=C["orange"])
        ax.text(5.9, 0.95 - 1.0, "backpropagation through time: gradients flow back along the chain of "
                                 "$\\mathbf{a}^{\\langle t \\rangle}$", ha="center", fontsize=13, color=C["red"])
        save(fig, "bptt")


def _mini(ax, n_in, n_out, kind, title, example):
    blank(ax, (-0.3, 6.2), (-1.2, 4.4))
    if kind == "one_to_one":
        circle(ax, 3.0, 0.4, "$x$", r=0.33, fs=11)
        box(ax, 3.0, 1.9, "", 0.7, 0.6)
        circle(ax, 3.0, 3.4, "$\\hat y$", r=0.33, fc="#FCE4EC", ec=C["red"], fs=11)
        arrow(ax, (3.0, 0.75), (3.0, 1.57))
        arrow(ax, (3.0, 2.22), (3.0, 3.05))
    else:
        T = max(n_in, 1) + (n_out if kind == "encdec" else 0) if kind == "encdec" else max(n_in, n_out)
        dx = 5.4 / max(T, 1)
        for t in range(T):
            xc = 0.4 + dx * (t + 0.5)
            box(ax, xc, 1.9, "", 0.55, 0.55)
            if t < T - 1:
                arrow(ax, (xc + 0.3, 1.9), (xc + dx - 0.3, 1.9), lw=1.3, ms=10)
            has_in = (kind in ("many_to_one", "many_to_many") or (kind == "one_to_many" and t == 0)
                      or (kind == "encdec" and t < n_in))
            has_out = (kind in ("one_to_many", "many_to_many") or (kind == "many_to_one" and t == T - 1)
                       or (kind == "encdec" and t >= n_in))
            if has_in:
                circle(ax, xc, 0.4, "", r=0.25)
                arrow(ax, (xc, 0.67), (xc, 1.6), lw=1.3, ms=10)
            if has_out:
                circle(ax, xc, 3.4, "", r=0.25, fc="#FCE4EC", ec=C["red"])
                arrow(ax, (xc, 2.2), (xc, 3.13), lw=1.3, ms=10)
            if kind == "one_to_many" and t > 0:
                arrow(ax, (xc - dx + 0.25, 3.4), (xc - 0.3, 2.15), lw=1.0, ms=9, ls="--", color=C["grey"])
    ax.text(3.0, 4.15, title, ha="center", fontsize=13, color=C["navy"])
    ax.text(3.0, -0.75, example, ha="center", fontsize=11, color=C["grey"])


def fig_rnn_types():
    with sketchy():
        fig, axs = plt.subplots(1, 5, figsize=(12.5, 3.2))
        fig.subplots_adjust(0.0, 0.0, 1.0, 1.0, wspace=0.05)
        _mini(axs[0], 1, 1, "one_to_one", "one-to-one", "standard network")
        _mini(axs[1], 1, 4, "one_to_many", "one-to-many", "music generation")
        _mini(axs[2], 4, 1, "many_to_one", "many-to-one", "sentiment analysis")
        _mini(axs[3], 4, 4, "many_to_many", "many-to-many", "named entities ($T_x = T_y$)")
        _mini(axs[4], 3, 3, "encdec", "encoder-decoder", "translation ($T_x \\neq T_y$)")
        save(fig, "rnn_types")


# --------------------------------------------------------------------------- 10b
def fig_language_model():
    words_in = ["$\\mathbf{0}$", "cats", "average", "15", "hours"]
    words_out = ["cats", "average", "15", "hours", "of"]
    for name, sample in [("lm_training", False), ("lm_sampling", True)]:
        with sketchy():
            fig, ax = plt.subplots(figsize=(12, 4.8))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (-0.4, 13.0), (-0.9, 4.6))
            pos = _unrolled(ax, 5, in_labels=words_in, out_labels=["$\\hat{\\mathbf{y}}^{\\langle %d \\rangle}$" % t
                                                                   for t in range(1, 6)], x0=1.6, dx=2.5)
            ax.text(12.6, 2.0, "...", fontsize=18, va="center", ha="center")
            for t, xc in enumerate(pos):
                if sample:
                    ax.text(xc, 4.25, "sample" if t == 0 else "sample", ha="center", fontsize=11, color=C["red"])
                    if t < 4:
                        arrow(ax, (xc + 0.38, 3.35), (pos[t + 1] - 0.3, 0.75), ls="--", color=C["red"], lw=1.5)
                else:
                    cond = "" if t == 0 else ("$\\mid$ " + " ".join(words_out[:t]) if t < 3 else "$\\mid$ ...")
                    ax.text(xc, 4.25, f"P({words_out[t]} {cond})", ha="center", fontsize=11, color=C["red"])
            note = ("training: the input at step $t$ is the true word $y^{\\langle t-1 \\rangle}$ (teacher forcing)"
                    if not sample else
                    "sampling: draw a word from $\\hat{\\mathbf{y}}^{\\langle t \\rangle}$ and feed it in as the next input")
            ax.text(6.4, -0.6, note, ha="center", fontsize=13, color=C["navy"])
            save(fig, name)


def _dino_data():
    import torch

    names = [w.strip().lower() for w in DATA.read_text().split() if w.strip()]
    chars = ["\n"] + sorted(set("".join(names)))            # index 0 = end of name
    idx = {c: i for i, c in enumerate(chars)}
    T = max(len(n) for n in names) + 1
    V = len(chars)
    X = torch.zeros(len(names), T, V)
    Y = torch.full((len(names), T), -100, dtype=torch.long)   # -100: ignored by CrossEntropyLoss
    for i, n in enumerate(names):
        seq = [idx[c] for c in n] + [0]
        for t, k in enumerate(seq):
            Y[i, t] = k
            if t + 1 < T:
                X[i, t + 1, k] = 1.0                           # input at t+1 = character at t (x<1> = 0)
    return names, chars, X, Y


def make_char_rnn(V, n_a=64):
    """A character-level RNN language model: one-hot input -> nn.RNN -> linear layer (logits)."""
    from torch import nn

    class CharRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(V, n_a, batch_first=True)
            self.out = nn.Linear(n_a, V)

        def forward(self, X, h=None):
            A, h = self.rnn(X, h)
            return self.out(A), h

    return CharRNN()


def _sample(model, chars, temperature=1.0, max_len=25, gen=None):
    import torch

    V = len(chars)
    x = torch.zeros(1, 1, V)
    h, name = None, ""
    with torch.no_grad():
        for _ in range(max_len):
            logits, h = model(x, h)
            p = torch.softmax(logits[0, -1] / temperature, dim=0)
            k = torch.multinomial(p, 1, generator=gen).item()
            if k == 0:
                break
            name += chars[k]
            x = torch.zeros(1, 1, V)
            x[0, 0, k] = 1.0
    return name


def fig_dinosaurs():
    import copy

    import torch
    from torch import nn

    names, chars, X, Y = _dino_data()
    perm = np.random.default_rng(0).permutation(len(names))
    n_val = len(names) // 5
    iv, it = perm[:n_val], perm[n_val:]                       # 20% of the names for validation
    torch.manual_seed(0)
    model = make_char_rnn(len(chars))
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    hist, best, best_state, best_it = [], np.inf, None, 0
    for i in range(600):
        logits, _ = model(X[it])
        loss = loss_fn(logits.reshape(-1, len(chars)), Y[it].reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            lv = loss_fn(model(X[iv])[0].reshape(-1, len(chars)), Y[iv].reshape(-1)).item()
        hist.append((loss.item(), lv))
        if lv < best:
            best, best_it, best_state = lv, i + 1, copy.deepcopy(model.state_dict())
    final = copy.deepcopy(model)
    model.load_state_dict(best_state)
    h = np.array(hist)
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        ep = np.arange(1, len(h) + 1)
        ax.plot(ep, h[:, 0], color=C["blue"], lw=2.6, label="training names")
        ax.plot(ep, h[:, 1], color=C["red"], lw=2.6, label="validation names")
        ax.axvline(best_it, color=C["grey"], ls=":", lw=1.5)
        ax.text(best_it + 8, 4.2, f"best validation loss\n(iteration {best_it})", fontsize=12, color=C["grey"])
        ax.set_ylim(0, 5)
        ax.set_xlabel("iteration (full batch)")
        ax.set_ylabel("cross-entropy per character")
        ax.legend(loc="lower left", fontsize=13, frameon=False)
        despine(ax)
        save(fig, "dino_loss")
    print(f"    {len(names)} names ({len(it)} train, {len(iv)} validation), {len(chars)} symbols")
    print(f"    best iteration {best_it}: train {h[best_it - 1, 0]:.3f}, val {best:.3f}; final: train {h[-1, 0]:.3f}, "
          f"val {h[-1, 1]:.3f}")
    for label, m in [("best", model), ("final", final)]:
        gen = torch.Generator().manual_seed(4)
        samples = [_sample(m, chars, 0.7, gen=gen) for _ in range(12)]
        novel = [x for x in samples if x not in set(names)]
        print(f"    {label}: {', '.join(x.capitalize() for x in samples)}  ({len(novel)}/12 not in the list)")


def fig_gradient_norms():
    """Vanilla RNN: size of d a<T> / d a<t> for recurrent weights of different spectral radii."""
    import torch

    torch.manual_seed(0)
    n_a, T = 32, 60
    W0 = torch.randn(n_a, n_a)
    W0 = W0 / torch.linalg.eigvals(W0).abs().max()
    x = 0.1 * torch.randn(T, n_a)
    res = {}
    for rho in (0.5, 0.9, 1.5):
        W = rho * W0
        a = torch.zeros(n_a)
        states = []
        for t in range(T):
            a = a.detach().requires_grad_(True) if t == 0 else a
            states.append(a)
            a = torch.tanh(W @ a + x[t])
        v = torch.randn(n_a)
        grads = torch.autograd.grad((a * v).sum(), states)
        res[rho] = [g.norm().item() for g in grads]
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        for rho, col in zip(res, [C["red"], C["orange"], C["green"]]):
            g = np.array(res[rho])
            dist = T - np.arange(T)
            ax.semilogy(dist, g, color=col, lw=2.6, label=f"$\\mathbf{{W}}_{{aa}}$ scaled to spectral radius {rho}")
        ax.set_xlabel("number of time steps back, $T - t$")
        ax.set_ylabel("$\\|\\partial L / \\partial \\mathbf{a}^{\\langle t \\rangle}\\|$ (log scale)")
        ax.legend(loc="lower left", fontsize=12, frameon=False)
        despine(ax)
        save(fig, "gradient_norms")
    for rho, g in res.items():
        print(f"    rho={rho}: 1 step back {g[-1]:.2e}, 20 steps {g[-20]:.2e}, 59 steps {g[0]:.2e}")


def fig_gradient_lstm():
    """Gradient that reaches the cell state c<t> of an LSTM, for a half-open and an open forget gate."""
    import torch
    from torch import nn

    n_x, n_a, T = 8, 32, 60
    torch.manual_seed(1)
    x = torch.randn(T, 1, n_x)
    res = {}
    for label, f_bias in [("LSTM, forget gate $\\approx 0.5$", 0.0), ("LSTM, forget gate $\\approx 0.99$", 5.0)]:
        torch.manual_seed(0)
        cell = nn.LSTMCell(n_x, n_a)
        with torch.no_grad():
            cell.bias_ih[n_a:2 * n_a] = f_bias                  # gates are stacked (i, f, g, o)
            cell.bias_hh[n_a:2 * n_a] = 0.0
        h = torch.zeros(1, n_a)
        c = torch.zeros(1, n_a)
        cs = []
        for t in range(T):
            c = c.detach().requires_grad_(True) if t == 0 else c
            cs.append(c)
            h, c = cell(x[t], (h, c))
        grads = torch.autograd.grad(h.sum(), cs)
        res[label] = [g.norm().item() for g in grads]
    with sketchy():
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        for (label, g), col in zip(res.items(), [C["orange"], C["green"]]):
            ax.semilogy(T - np.arange(T), g, color=col, lw=2.6, label=label)
        ax.set_xlabel("number of time steps back, $T - t$")
        ax.set_ylabel("$\\|\\partial L / \\partial \\mathbf{c}^{\\langle t \\rangle}\\|$ (log scale)")
        ax.legend(loc="lower left", fontsize=12, frameon=False)
        despine(ax)
        save(fig, "gradient_lstm")
    for label, g in res.items():
        print(f"    {label}: 1 step back {g[-1]:.2e}, 20 steps {g[-20]:.2e}, 59 steps {g[0]:.2e}")


# --------------------------------------------------------------------------- 10c
def _gate(ax, x, y, text, fc="#FFF8E1", ec=None, w=1.5, h=0.6, fs=12):
    box(ax, x, y, text, w, h, fc=fc, ec=ec or C["orange"], fs=fs)


def fig_gru():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 14), (-0.6, 5.4))
        ax.text(0.8, 4.3, "$\\mathbf{c}^{\\langle t-1 \\rangle}$", fontsize=16, ha="center", va="center")
        arrow(ax, (1.3, 4.3), (9.4, 4.3), lw=2.4, color=C["navy"])
        circle(ax, 9.8, 4.3, "mix", r=0.42, fc="white", ec=C["navy"], fs=12)
        arrow(ax, (10.22, 4.3), (12.5, 4.3), lw=2.4, color=C["navy"])
        ax.text(13.3, 4.3, "$\\mathbf{c}^{\\langle t \\rangle} = \\mathbf{a}^{\\langle t \\rangle}$", fontsize=14,
                ha="center", va="center")
        ax.text(5.0, 0.0, "$[\\mathbf{c}^{\\langle t-1 \\rangle}, \\mathbf{x}^{\\langle t \\rangle}]$", fontsize=14,
                ha="center")
        _gate(ax, 3.2, 2.0, "$\\Gamma_r = \\sigma(\\ldots)$", w=1.9)
        _gate(ax, 6.0, 2.0, "$\\Gamma_u = \\sigma(\\ldots)$", w=1.9)
        _gate(ax, 8.6, 2.0, "$\\tilde{\\mathbf{c}} = \\tanh(\\ldots)$", fc="#E3F2FD", ec=C["blue"], w=2.1)
        for xg in (3.2, 6.0):
            arrow(ax, (5.0, 0.45), (xg, 1.67), lw=1.3)
        arrow(ax, (5.0, 0.45), (8.3, 1.67), lw=1.3)
        arrow(ax, (4.17, 2.0), (7.52, 2.0), lw=1.3, rad=-0.25)
        ax.text(5.9, 1.05, "reset: how much of $\\mathbf{c}^{\\langle t-1 \\rangle}$ enters $\\tilde{\\mathbf{c}}$",
                fontsize=11, color=C["grey"], ha="center")
        arrow(ax, (6.0, 2.32), (9.45, 4.0), lw=1.5, rad=0.1)
        arrow(ax, (8.6, 2.32), (9.7, 3.88), lw=1.5)
        ax.text(10.4, 2.9, "$\\mathbf{c}^{\\langle t \\rangle} = \\Gamma_u \\odot \\tilde{\\mathbf{c}} + "
                           "(1 - \\Gamma_u) \\odot \\mathbf{c}^{\\langle t-1 \\rangle}$", fontsize=12, color=C["navy"])
        ax.text(5.0, 4.75, "memory line: copied through unless the update gate $\\Gamma_u$ opens", fontsize=12,
                color=C["navy"], ha="center")
        save(fig, "gru")


def fig_lstm():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11.5, 5.0))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 14), (-0.6, 5.6))
        ax.text(0.8, 4.4, "$\\mathbf{c}^{\\langle t-1 \\rangle}$", fontsize=16, ha="center", va="center")
        arrow(ax, (1.3, 4.4), (3.6, 4.4), lw=2.4, color=C["navy"])
        circle(ax, 4.0, 4.4, "$\\odot$", r=0.35, fc="white", ec=C["navy"], fs=15)
        arrow(ax, (4.35, 4.4), (7.0, 4.4), lw=2.4, color=C["navy"])
        circle(ax, 7.4, 4.4, "$+$", r=0.35, fc="white", ec=C["navy"], fs=15)
        arrow(ax, (7.75, 4.4), (12.5, 4.4), lw=2.4, color=C["navy"])
        ax.text(13.2, 4.4, "$\\mathbf{c}^{\\langle t \\rangle}$", fontsize=16, ha="center", va="center")
        ax.text(1.0, 0.0, "$\\mathbf{a}^{\\langle t-1 \\rangle}, \\mathbf{x}^{\\langle t \\rangle}$", fontsize=14,
                ha="center")
        gates = [(4.0, "$\\Gamma_f$: forget", "#FFF8E1", C["orange"]), (6.4, "$\\Gamma_u$: update", "#FFF8E1", C["orange"]),
                 (8.5, "$\\tilde{\\mathbf{c}} = \\tanh(\\ldots)$", "#E3F2FD", C["blue"]),
                 (10.8, "$\\Gamma_o$: output", "#FFF8E1", C["orange"])]
        for xg, t, fc, ec in gates:
            _gate(ax, xg, 1.9, t, fc=fc, ec=ec, w=1.9)
            arrow(ax, (1.8, 0.25), (xg - 0.3, 1.57), lw=1.1)
        arrow(ax, (4.0, 2.22), (4.0, 4.05), lw=1.5)
        circle(ax, 7.4, 3.1, "$\\odot$", r=0.3, fc="white", ec="k", fs=14)
        arrow(ax, (6.4, 2.22), (7.15, 2.95), lw=1.4)
        arrow(ax, (8.5, 2.22), (7.65, 2.95), lw=1.4)
        arrow(ax, (7.4, 3.4), (7.4, 4.05), lw=1.5)
        box(ax, 11.6, 3.35, "tanh", 0.9, 0.5, fc="white", ec="k", fs=12)
        arrow(ax, (11.6, 4.4), (11.6, 3.62), lw=1.4)
        circle(ax, 11.6, 2.6, "$\\odot$", r=0.28, fc="white", ec="k", fs=13)
        arrow(ax, (11.6, 3.1), (11.6, 2.88), lw=1.4)
        arrow(ax, (10.8, 2.22), (11.35, 2.5), lw=1.4)
        arrow(ax, (11.88, 2.6), (12.9, 2.6), lw=2.0)
        ax.text(13.4, 2.6, "$\\mathbf{a}^{\\langle t \\rangle}$", fontsize=16, ha="center", va="center")
        ax.text(6.8, 5.15, "cell state: $\\mathbf{c}^{\\langle t \\rangle} = \\Gamma_f \\odot \\mathbf{c}^{\\langle t-1 \\rangle} "
                           "+ \\Gamma_u \\odot \\tilde{\\mathbf{c}}$", fontsize=13, color=C["navy"], ha="center")
        save(fig, "lstm")


def fig_bidirectional():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.2, 12.4), (-0.3, 5.4))
        pos = [1.8 + 2.8 * t for t in range(4)]
        for t, xc in enumerate(pos, start=1):
            circle(ax, xc, 0.3, f"$\\mathbf{{x}}^{{\\langle {t} \\rangle}}$")
            box(ax, xc, 1.6, f"$\\overrightarrow{{\\mathbf{{a}}}}^{{\\langle {t} \\rangle}}$", 1.0, 0.7, fs=13)
            box(ax, xc, 2.9, f"$\\overleftarrow{{\\mathbf{{a}}}}^{{\\langle {t} \\rangle}}$", 1.0, 0.7, fc="#FFF3E0",
                ec=C["orange"], fs=13)
            circle(ax, xc, 4.5, f"$\\hat{{\\mathbf{{y}}}}^{{\\langle {t} \\rangle}}$", fc="#FCE4EC", ec=C["red"])
            arrow(ax, (xc, 0.68), (xc, 1.23))
            arrow(ax, (xc - 0.25, 0.68), (xc - 0.4, 2.53), rad=-0.3, lw=1.2)
            arrow(ax, (xc + 0.3, 1.95), (xc + 0.2, 4.1), rad=0.35, lw=1.2)
            arrow(ax, (xc, 3.27), (xc, 4.1))
            if t < 4:
                arrow(ax, (xc + 0.52, 1.6), (xc + 2.28, 1.6), color=C["blue"])
                arrow(ax, (xc + 2.28, 2.9), (xc + 0.52, 2.9), color=C["orange"])
        ax.text(11.6, 1.6, "forward", fontsize=12, color=C["blue"], va="center")
        ax.text(11.6, 2.9, "backward", fontsize=12, color=C["orange"], va="center")
        save(fig, "bidirectional")


def fig_deep_rnn():
    with sketchy():
        fig, ax = plt.subplots(figsize=(11, 5.2))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.6, 12.2), (-0.4, 6.4))
        pos = [1.8 + 2.7 * t for t in range(4)]
        for t, xc in enumerate(pos, start=1):
            circle(ax, xc, 0.2, f"$\\mathbf{{x}}^{{\\langle {t} \\rangle}}$")
            arrow(ax, (xc, 0.58), (xc, 1.08))
            for l in range(1, 4):
                yc = 1.4 * l
                box(ax, xc, yc, f"$\\mathbf{{a}}^{{[{l}]\\langle {t} \\rangle}}$", 1.2, 0.65, fs=12)
                if l < 3:
                    arrow(ax, (xc, yc + 0.35), (xc, yc + 1.05))
                if t < 4:
                    arrow(ax, (xc + 0.62, yc), (xc + 2.08, yc))
            circle(ax, xc, 5.75, f"$\\hat{{\\mathbf{{y}}}}^{{\\langle {t} \\rangle}}$", fc="#FCE4EC", ec=C["red"])
            arrow(ax, (xc, 4.55), (xc, 5.37))
        for l in range(1, 4):
            ax.text(-0.3, 1.4 * l, f"layer {l}", fontsize=12, va="center", color=C["navy"])
        save(fig, "deep_rnn")


def fig_embedding_map():
    pts = {"man": (1.0, 1.0), "woman": (1.0, 2.4), "king": (3.4, 1.3), "queen": (3.4, 2.7),
           "boy": (0.3, 0.6), "girl": (0.3, 2.0),
           "apple": (6.6, 3.9), "orange": (7.2, 3.4), "banana": (6.9, 4.4), "grape": (7.6, 4.1),
           "one": (6.4, 0.6), "two": (7.0, 0.9), "three": (7.5, 0.5), "four": (7.1, 0.2),
           "cat": (4.6, 4.4), "dog": (5.0, 4.0), "horse": (4.3, 3.8)}
    with sketchy():
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
        ax.set_xlim(-0.4, 8.6)
        ax.set_ylim(-0.4, 5.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for w, (x, y) in pts.items():
            ax.scatter([x], [y], s=40, color=C["blue"], zorder=4)
            ax.text(x + 0.1, y + 0.1, w, fontsize=13)
        arrow(ax, pts["man"], pts["woman"], color=C["red"], lw=2)
        arrow(ax, pts["king"], pts["queen"], color=C["red"], lw=2)
        ax.text(2.0, 1.85, "same direction:\nmale $\\to$ female", fontsize=12, color=C["red"])
        ax.text(8.4, -0.25, "illustration: a 2D projection of word vectors", ha="right", fontsize=11, color=C["grey"])
        despine(ax)
        save(fig, "embedding_map")


FIGURES = {
    "one_hot": fig_one_hot, "dense_on_sequence": fig_dense_on_sequence, "unrolled": fig_unrolled,
    "rnn_cell": fig_rnn_cell, "bptt": fig_bptt, "rnn_types": fig_rnn_types, "language_model": fig_language_model,
    "dinosaurs": fig_dinosaurs, "gradient_norms": fig_gradient_norms, "gradient_lstm": fig_gradient_lstm,
    "gru": fig_gru, "lstm": fig_lstm, "bidirectional": fig_bidirectional, "deep_rnn": fig_deep_rnn,
    "embedding_map": fig_embedding_map,
}

if __name__ == "__main__":
    wanted = sys.argv[1:]
    print(f"Font: {FONTS[0]}  ->  {OUT}")
    for key, fn in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            print(key)
            fn()
