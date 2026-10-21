"""
Sketch-style figures for Lecture 8: Convolutional neural networks.
    part 8a   convolution, edge detection, padding and stride
    part 8b   convolutions over volumes, pooling, CNN architectures
    part 8c   transfer learning, data augmentation, face verification, neural style transfer

Run from anywhere:
    python scripts/make_figures_lecture08.py                 # all figures
    python scripts/make_figures_lecture08.py conv pool       # only figures whose name contains these words

Output: figures/cnn-sketch/*.png (relative to the project root).

Requirements: numpy, scipy, matplotlib >= 3.7, scikit-learn (sample images, 8 x 8 digits), Pillow;
PyTorch for the figures with trained networks, torchvision for the augmentation figure and for
neural style transfer (fixed seeds, so the numbers quoted on the slides are reproducible).
The photographs are scikit-learn's bundled sample images (sklearn.datasets.load_sample_image) and the
University cloisters (styles/backgrounds/source/title-photo.jpg); the style images are generated here.

Neural style transfer (figure key "nst") downloads the pretrained VGG-19 once (about 550 MB) and runs
300 L-BFGS iterations per example (images of 256 pixels on a CPU, 384 on a GPU): about five
minutes on a laptop CPU, under a minute on a GPU. Run it on its own with
    python scripts/make_figures_lecture08.py nst
Its figures are only computed if they do not exist yet, so later runs of the script skip it. To compute
them again (e.g. after changing style_weight), add --force:
    python scripts/make_figures_lecture08.py nst --force

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
OUT = ROOT / "figures" / "cnn-sketch"
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
from scipy.signal import correlate2d  # noqa: E402
from sklearn.datasets import load_digits, load_sample_image  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

X_NG = np.array([[3, 0, 1, 2, 7, 4],
                 [1, 5, 8, 9, 3, 1],
                 [2, 7, 2, 5, 1, 3],
                 [0, 1, 3, 1, 7, 8],
                 [4, 2, 1, 6, 2, 8],
                 [2, 4, 5, 2, 3, 9]])
K_VERT = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])


def box(ax, xc, yc, text, w, h, fc="#FCE4EC", ec="k", fs=15, lw=2, color="k"):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, p, q, color="k", lw=2.0, ms=16, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=ms,
                                                    connectionstyle=f"arc3,rad={rad}"))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def grid(ax, M, x0, y0, cell=1.0, fc="white", fmt="{:g}", fs=13, hl=None, hl_color=None, shade=None, vmax=None,
         hide=None, lw=1.4):
    """Draw matrix M with its top-left corner at (x0, y0). hl = (row, col, h, w) highlight rectangle."""
    rows, cols = M.shape
    for r in range(rows):
        for c in range(cols):
            col = fc
            if shade is not None:
                v = shade[r, c] / (vmax or max(1e-9, np.abs(shade).max()))
                g = 1 - 0.45 * np.clip(v, 0, 1)            # light shades keep the dark digits readable
                col = (g, g, g)
            ax.add_patch(plt.Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell, fc=col, ec="k", lw=lw))
            if hide is not None and hide[r, c]:
                continue
            ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, fmt.format(M[r, c]), ha="center", va="center",
                    fontsize=fs, color="k")
    if hl is not None:
        r, c, h, w = hl
        ax.add_patch(plt.Rectangle((x0 + c * cell, y0 - (r + h) * cell), w * cell, h * cell, fill=False,
                                   ec=hl_color or C["red"], lw=3.2, zorder=6))


def load_gray(name="china.jpg"):
    return load_sample_image(name).mean(axis=2) / 255.0


# --------------------------------------------------------------------------- 9a
def fig_cv_tasks():
    flower = load_sample_image("flower.jpg")
    china = load_sample_image("china.jpg")
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88, wspace=0.05)
        axs[0].imshow(flower)
        axs[0].set_title("classification: is this a flower? (0/1)", fontsize=15)
        axs[1].imshow(china)
        axs[1].add_patch(plt.Rectangle((95, 25), 400, 300, fill=False, ec=C["red"], lw=3))
        axs[1].text(100, 55, "building", color="k", fontsize=15, bbox=dict(fc="white", ec=C["red"], lw=1.5))
        axs[1].set_title("object detection: what is where?", fontsize=15)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        save(fig, "cv_tasks")


def fig_sobel():
    g = load_gray("china.jpg")
    sobel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gv = np.abs(correlate2d(g, sobel_v, mode="valid"))
    gh = np.abs(correlate2d(g, sobel_v.T, mode="valid"))
    with sketchy():
        fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86, wspace=0.05)
        for ax, im, t in [(axs[0], g, "greyscale image"), (axs[1], gv, "vertical edges (Sobel)"),
                          (axs[2], gh, "horizontal edges (Sobel)")]:
            ax.imshow(im, cmap="gray_r" if im is not g else "gray", vmax=None if im is g else np.quantile(im, 0.99))
            ax.set_title(t, fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
        save(fig, "sobel")


def fig_conv_steps():
    out = correlate2d(X_NG, K_VERT, mode="valid")
    positions = [(0, 0), (0, 1), (3, 3)]
    for step, (r, c) in enumerate(positions, start=1):
        shown = np.zeros_like(out, dtype=bool)
        if step == 1:
            shown[0, 0] = True
        elif step == 2:
            shown[0, :2] = True
        else:
            shown[:] = True
        with sketchy():
            fig, ax = plt.subplots(figsize=(12, 4.6))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (-0.3, 16.6), (-0.6, 6.9))
            grid(ax, X_NG, 0, 6.3, hl=(r, c, 3, 3))
            ax.text(3, -0.3, "input $6 \\times 6$", ha="center", fontsize=14)
            ax.text(6.7, 3.3, "$\\ast$", fontsize=26, ha="center", va="center")
            grid(ax, K_VERT, 7.5, 4.8, fc="#FFF8E1")
            ax.text(9.0, 1.2, "filter $3 \\times 3$", ha="center", fontsize=14)
            ax.text(11.2, 3.3, "$=$", fontsize=24, ha="center", va="center")
            grid(ax, out, 12.0, 5.3, fc="#FCE4EC", hide=~shown, hl=(r, c, 1, 1) if step < 3 else None)
            ax.text(14.0, 0.6, "output $4 \\times 4$", ha="center", fontsize=14)
            save(fig, f"conv_step{step}")
    print("    conv output:\n", out)


def fig_edge_example():
    left = np.array([[10, 10, 10, 0, 0, 0]] * 6)
    with sketchy():
        fig, axs = plt.subplots(2, 1, figsize=(11, 6.6))
        fig.subplots_adjust(0, 0, 1, 1, hspace=0.05)
        for ax, X, t in [(axs[0], left, "bright to dark"), (axs[1], left[:, ::-1], "dark to bright")]:
            blank(ax, (-0.3, 17.5), (-0.4, 6.6))
            out = correlate2d(X, K_VERT, mode="valid")
            grid(ax, X, 0, 6.3, cell=0.95, shade=X, vmax=10, fs=12)
            ax.text(6.4, 3.3, "$\\ast$", fontsize=24, ha="center", va="center")
            grid(ax, K_VERT, 7.1, 4.75, cell=0.95, shade=K_VERT + 1, vmax=2, fs=12)
            ax.text(10.6, 3.3, "$=$", fontsize=22, ha="center", va="center")
            grid(ax, out, 11.3, 5.25, cell=0.95, shade=out - out.min(), vmax=max(1, np.ptp(out)), fs=12)
            ax.text(16.0, 3.0, t, fontsize=15, ha="left", va="center", color=C["navy"], rotation=90)
        save(fig, "edge_example")


def fig_filters():
    filters = [("vertical", K_VERT), ("horizontal", K_VERT.T),
               ("Sobel", np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])),
               ("Scharr", np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]))]
    with sketchy():
        fig, ax = plt.subplots(figsize=(12, 3.2))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.3, 17.6), (-0.9, 3.9))
        for k, (name, K) in enumerate(filters):
            x0 = 0.4 + 4.4 * k
            grid(ax, K, x0, 3.2, fc="#FFF8E1")
            ax.text(x0 + 1.5, -0.45, name, ha="center", fontsize=15)
        save(fig, "filters")


def _digits_data():
    import torch

    d = load_digits()
    X = torch.tensor(d.images / 16.0, dtype=torch.float32).unsqueeze(1)        # (1797, 1, 8, 8)
    y = torch.tensor(d.target)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=d.target)
    return X, y, idx_tr, idx_te


def _small_cnn():
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8 x 4 x 4
        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 16 x 2 x 2
        nn.Flatten(), nn.Linear(64, 10))


def _mlp():
    from torch import nn

    return nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def _train(model, X, y, idx_tr, idx_te, epochs=40, lr=0.1, seed=0):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X[idx_tr], y[idx_tr]), batch_size=32, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    accs = []
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            loss = loss_fn(model(Xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            accs.append((model(X[idx_te]).argmax(1) == y[idx_te]).float().mean().item())
    return accs


def fig_learned_filters_and_cnn_vs_mlp():
    import torch

    X, y, idx_tr, idx_te = _digits_data()
    torch.manual_seed(0)
    cnn = _small_cnn()
    acc_cnn = _train(cnn, X, y, idx_tr, idx_te)
    torch.manual_seed(0)
    mlp = _mlp()
    acc_mlp = _train(mlp, X, y, idx_tr, idx_te)
    n_cnn = sum(p.numel() for p in cnn.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    W = cnn[0].weight.detach().numpy()[:, 0]                  # (8, 3, 3)
    x0 = X[idx_te[0]:idx_te[0] + 1]
    with torch.no_grad():
        fmap = torch.relu(cnn[0](x0))[0].numpy()             # (8, 8, 8)
    with sketchy():
        fig, axs = plt.subplots(2, 9, figsize=(12.5, 3.6), gridspec_kw=dict(width_ratios=[1.3] + [1] * 8))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.86, wspace=0.12, hspace=0.3)
        axs[0, 0].axis("off")
        axs[0, 0].text(0.5, 0.5, "learned\n3 x 3\nfilters", ha="center", va="center", fontsize=13,
                       transform=axs[0, 0].transAxes)
        axs[1, 0].imshow(X[idx_te[0], 0].numpy(), cmap="gray_r")
        axs[1, 0].set_title("input", fontsize=12)
        for k in range(8):
            v = np.abs(W[k]).max()
            axs[0, k + 1].imshow(W[k], cmap="RdBu_r", vmin=-v, vmax=v)
            axs[1, k + 1].imshow(fmap[k], cmap="gray_r")
            axs[0, k + 1].set_title(f"filter {k + 1}", fontsize=11)
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.text(0.5, 0.94, "top: the 8 filters of the first layer after training; bottom: their feature maps "
                            "(after ReLU) for one test digit", ha="center", fontsize=13)
        save(fig, "learned_filters")
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.95)
        ep = np.arange(1, len(acc_cnn) + 1)
        ax.plot(ep, 100 * np.array(acc_mlp), color=C["orange"], lw=2.8, label=f"MLP, {n_mlp:,} parameters")
        ax.plot(ep, 100 * np.array(acc_cnn), color=C["blue"], lw=2.8, label=f"small CNN, {n_cnn:,} parameters")
        ax.set_ylim(0, 102)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy (%)")
        ax.legend(loc="lower right", fontsize=13, frameon=False)
        despine(ax)
        save(fig, "cnn_vs_mlp")
    print(f"    CNN {n_cnn} params, final acc {acc_cnn[-1]:.4f}; MLP {n_mlp} params, final acc {acc_mlp[-1]:.4f}")


def fig_padding():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 10, size=(6, 6))
    P = np.pad(X, 1)
    pad_mask = np.ones_like(P, dtype=bool)
    pad_mask[1:-1, 1:-1] = False
    with sketchy():
        fig, ax = plt.subplots(figsize=(11, 4.8))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (-0.3, 16.6), (-1.0, 8.3))
        for r in range(8):
            for c in range(8):
                ax.add_patch(plt.Rectangle((c, 8 - r - 1), 1, 1, fc="#E0E0E0" if pad_mask[r, c] else "white",
                                           ec="k", lw=1.2))
                ax.text(c + 0.5, 8 - r - 0.5, str(P[r, c]), ha="center", va="center", fontsize=12,
                        color=C["grey"] if pad_mask[r, c] else "k")
        ax.add_patch(plt.Rectangle((0, 5), 3, 3, fill=False, ec=C["red"], lw=3.2))
        ax.text(4, -0.6, "$6 \\times 6$ input, padded with $p = 1$: $8 \\times 8$", ha="center", fontsize=14)
        ax.text(9.9, 4.0, "$\\ast$ filter $3\\times 3$ $=$", fontsize=16, ha="center", va="center")
        grid(ax, np.zeros((6, 6)), 11.6, 6.4, fc="#FCE4EC", hide=np.ones((6, 6), bool), cell=0.8)
        ax.text(14.0, 1.0, "output $6 \\times 6$", ha="center", fontsize=14)
        save(fig, "padding")


def fig_stride_steps():
    rng = np.random.default_rng(5)
    X = rng.integers(0, 10, size=(7, 7))
    K = np.array([[3, 4, 4], [1, 0, 2], [-1, 0, 3]])
    out = correlate2d(X, K, mode="valid")[::2, ::2]
    positions = [(0, 0), (0, 2), (2, 0)]
    for step, (r, c) in enumerate(positions, start=1):
        shown = np.zeros_like(out, dtype=bool)
        if step == 1:
            shown[0, 0] = True
        elif step == 2:
            shown[0, :2] = True
        else:
            shown[:] = True
        with sketchy():
            fig, ax = plt.subplots(figsize=(12, 4.6))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (-0.3, 16.6), (-0.8, 7.4))
            grid(ax, X, 0, 7.0, cell=0.95, hl=(r, c, 3, 3))
            ax.text(3.3, -0.45, "input $7 \\times 7$, stride $s = 2$", ha="center", fontsize=14)
            ax.text(7.4, 3.6, "$\\ast$", fontsize=26, ha="center", va="center")
            grid(ax, K, 8.1, 5.0, fc="#FFF8E1")
            ax.text(9.6, 1.4, "filter $3 \\times 3$", ha="center", fontsize=14)
            ax.text(11.8, 3.6, "$=$", fontsize=24, ha="center", va="center")
            grid(ax, out, 12.6, 5.1, fc="#FCE4EC", hide=~shown, hl=(r // 2, c // 2, 1, 1) if step < 3 else None)
            ax.text(14.1, 1.4, "output $3 \\times 3$", ha="center", fontsize=14)
            save(fig, f"stride_step{step}")
    print("    stride input\n", X, "\n    stride output\n", out)


# --------------------------------------------------------------------------- 9b
def _cube(ax, x0, y0, n, depth, cell, colors, labels=None, alpha=1.0):
    """Draw `depth` stacked n x n grids (back to front), offset diagonally."""
    off = 0.35 * cell
    for d in range(depth - 1, -1, -1):
        dx, dy = d * off, d * off
        for r in range(n):
            for c in range(n):
                ax.add_patch(plt.Rectangle((x0 + dx + c * cell, y0 + dy - (r + 1) * cell), cell, cell,
                                           fc=colors[d], ec="k", lw=0.9, alpha=alpha))
        if labels:
            ax.text(x0 + dx + n * cell + 0.1, y0 + dy - 0.3 * cell, labels[d], fontsize=11, va="center")


def fig_volume():
    rgb = ["#FFCDD2", "#C8E6C9", "#BBDEFB"]
    for name, n_filters in [("volume_one_filter", 1), ("volume_two_filters", 2)]:
        with sketchy():
            fig, ax = plt.subplots(figsize=(12, 4.6))
            fig.subplots_adjust(0, 0, 1, 1)
            blank(ax, (0, 16.4), (-1.2, 6.0))
            _cube(ax, 0.2, 4.4, 6, 3, 0.62, rgb, labels=["R", "G", "B"][::-1] and ["R", "G", "B"])
            ax.text(2.1, -0.4, "input $6 \\times 6 \\times 3$", ha="center", fontsize=14)
            fcols = [["#FFE0B2"] * 3, ["#E1BEE7"] * 3]
            outs = ["#FFE0B2", "#E1BEE7"]
            for k in range(n_filters):
                yk = 4.2 - 2.6 * k if n_filters == 2 else 3.4
                ax.text(5.6, yk - 0.9, "$\\ast$", fontsize=22, ha="center", va="center")
                _cube(ax, 6.2, yk, 3, 3, 0.55, fcols[k])
                ax.text(7.4, yk - 2.05, f"filter {k + 1}: $3 \\times 3 \\times 3$", ha="center", fontsize=12)
                arrow(ax, (8.9, yk - 0.9), (10.2, yk - 0.9))
                grid(ax, np.zeros((4, 4)), 10.4, yk + 0.35, cell=0.5, fc=outs[k], hide=np.ones((4, 4), bool))
                ax.text(11.4, yk - 1.9, "$4 \\times 4$", ha="center", fontsize=12)
            if n_filters == 2:
                arrow(ax, (12.6, 2.2), (13.4, 2.2))
                _cube(ax, 13.6, 3.2, 4, 2, 0.5, outs[::-1])
                ax.text(14.7, 0.3, "output $4 \\times 4 \\times 2$", ha="center", fontsize=13)
            save(fig, name)


def fig_pooling():
    X = np.array([[1, 3, 2, 1], [2, 9, 1, 1], [1, 3, 2, 3], [5, 6, 1, 2]])
    blocks = X.reshape(2, 2, 2, 2).swapaxes(1, 2)
    mx = blocks.max(axis=(2, 3))
    av = blocks.mean(axis=(2, 3))
    cols = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0"]
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
        fig.subplots_adjust(0.01, 0.02, 0.99, 0.86, wspace=0.05)
        for ax, M, t, fmt in [(axs[0], mx, "max pooling ($f = 2$, $s = 2$)", "{:g}"),
                              (axs[1], av, "average pooling ($f = 2$, $s = 2$)", "{:.2f}")]:
            blank(ax, (-0.2, 8.6), (-0.4, 4.4))
            for r in range(4):
                for c in range(4):
                    ax.add_patch(plt.Rectangle((c, 4 - r - 1), 1, 1, fc=cols[(r // 2) * 2 + c // 2], ec="k", lw=1.2))
                    ax.text(c + 0.5, 4 - r - 0.5, str(X[r, c]), ha="center", va="center", fontsize=14)
            arrow(ax, (4.3, 2.0), (5.3, 2.0))
            for r in range(2):
                for c in range(2):
                    ax.add_patch(plt.Rectangle((5.6 + c * 1.2, 3.2 - (r + 1) * 1.2), 1.2, 1.2, fc=cols[r * 2 + c],
                                               ec="k", lw=1.4))
                    ax.text(5.6 + (c + 0.5) * 1.2, 3.2 - (r + 0.5) * 1.2, fmt.format(M[r, c]), ha="center",
                            va="center", fontsize=14)
            ax.set_title(t, fontsize=15)
        save(fig, "pooling")


def _block_diagram(name, layers, title=None, width=12.5):
    """layers: list of (label, shape_text, kind) with kind in {in, conv, pool, fc, out}."""
    styles = {"in": ("#F5F5F5", "k"), "conv": ("#E3F2FD", C["blue"]), "pool": ("#FFF8E1", C["orange"]),
              "fc": ("#E8F5E9", C["green"]), "out": ("#FCE4EC", C["red"])}
    n = len(layers)
    with sketchy():
        fig, ax = plt.subplots(figsize=(width, 2.9))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 1.6 * n + 0.2), (-0.9, 2.2))
        for k, (lab, shp, kind) in enumerate(layers):
            xc = 0.9 + 1.6 * k
            fc, ec = styles[kind]
            box(ax, xc, 1.0, lab, 1.25, 1.0, fc=fc, ec=ec, fs=12)
            ax.text(xc, -0.25, shp, ha="center", fontsize=11, color=C["navy"])
            if k < n - 1:
                arrow(ax, (xc + 0.66, 1.0), (xc + 0.95, 1.0), lw=1.6, ms=12)
        if title:
            ax.text(0.8 * n + 0.1, 1.95, title, ha="center", fontsize=13, color=C["navy"])
        save(fig, name)


def fig_architectures():
    _block_diagram("example_net", [
        ("input", "32 x 32 x 3", "in"), ("CONV1\n5 x 5, 8", "28 x 28 x 8", "conv"), ("POOL1\nmax 2", "14 x 14 x 8", "pool"),
        ("CONV2\n5 x 5, 16", "10 x 10 x 16", "conv"), ("POOL2\nmax 2", "5 x 5 x 16", "pool"), ("flatten", "400", "in"),
        ("FC3", "120", "fc"), ("FC4", "84", "fc"), ("softmax", "10", "out")])
    _block_diagram("lenet5", [
        ("input", "32 x 32 x 1", "in"), ("conv\n5 x 5, 6", "28 x 28 x 6", "conv"), ("avg pool\n2, s = 2", "14 x 14 x 6", "pool"),
        ("conv\n5 x 5, 16", "10 x 10 x 16", "conv"), ("avg pool\n2, s = 2", "5 x 5 x 16", "pool"),
        ("conv / FC\n120", "120", "fc"), ("FC", "84", "fc"), ("output", "10", "out")])


# --------------------------------------------------------------------------- 9c
def fig_transfer():
    with sketchy():
        fig, ax = plt.subplots(figsize=(12, 4.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 15.5), (-0.6, 6.0))
        ax.text(0.2, 5.4, "task 1: ImageNet, 1.2 million images, 1000 classes", fontsize=14, color=C["navy"])
        box(ax, 1.3, 4.2, "image", 1.8, 0.9, fc="#F5F5F5")
        box(ax, 6.2, 4.2, "convolutional layers (body)", 6.4, 1.0, fc="#E3F2FD", ec=C["blue"], fs=14)
        box(ax, 11.6, 4.2, "head: 1000 classes", 3.0, 1.0, fc="#FCE4EC", ec=C["red"], fs=14)
        arrow(ax, (2.25, 4.2), (2.95, 4.2))
        arrow(ax, (9.45, 4.2), (10.05, 4.2))
        ax.text(0.2, 2.4, "task 2: a few hundred images of bees and ants", fontsize=14, color=C["navy"])
        box(ax, 1.3, 1.2, "image", 1.8, 0.9, fc="#F5F5F5")
        box(ax, 6.2, 1.2, "same body: copied weights", 6.4, 1.0, fc="#E3F2FD", ec=C["blue"], fs=14)
        box(ax, 11.6, 1.2, "new head: 2 classes", 3.0, 1.0, fc="#E8F5E9", ec=C["green"], fs=14)
        arrow(ax, (2.25, 1.2), (2.95, 1.2))
        arrow(ax, (9.45, 1.2), (10.05, 1.2))
        arrow(ax, (6.2, 3.65), (6.2, 1.75), color=C["blue"], lw=2.4)
        ax.text(6.4, 2.65, "reuse", fontsize=13, color=C["blue"])
        ax.text(6.2, 0.2, "frozen, or fine-tuned with a small learning rate", ha="center", fontsize=12, color=C["grey"])
        ax.text(11.6, 0.2, "trained from scratch", ha="center", fontsize=12, color=C["grey"])
        save(fig, "transfer")


def fig_augment():
    import torch
    from PIL import Image
    from torchvision.transforms import v2

    img = Image.fromarray(load_sample_image("flower.jpg"))
    transforms = [("original", None),
                  ("RandomResizedCrop", v2.RandomResizedCrop(size=(300, 400), scale=(0.3, 0.8), antialias=True)),
                  ("RandomHorizontalFlip", v2.RandomHorizontalFlip(p=1.0)),
                  ("ColorJitter", v2.ColorJitter(brightness=0.5, hue=0.3)),
                  ("RandomRotation", v2.RandomRotation(degrees=30)),
                  ("RandomPerspective", v2.RandomPerspective(distortion_scale=0.5, p=1.0))]
    torch.manual_seed(3)
    with sketchy():
        fig, axs = plt.subplots(2, 3, figsize=(12, 5.6))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.94, wspace=0.05, hspace=0.18)
        for ax, (name, t) in zip(axs.ravel(), transforms):
            out = img if t is None else t(img)
            ax.imshow(out)
            ax.set_title(name, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        save(fig, "augment")


PEOPLE = [dict(skin="#FFE0BD", hair=C["navy"], style="spiky"),
          dict(skin="#F1C27D", hair="#8D5524", style="bob"),
          dict(skin="#FFDBAC", hair="#E6A01E", style="flat"),
          dict(skin="#C68642", hair="k", style="bald")]


def draw_face(ax, x, y, s, person, variant=0):
    """A simple cartoon face of `person` (0..3); `variant` changes expression and tilt slightly."""
    p = PEOPLE[person]
    tilt = [0, 8, -8][variant % 3]
    from matplotlib.patches import Ellipse, Arc

    ax.add_patch(Ellipse((x, y), 1.6 * s, 2.0 * s, angle=tilt, fc=p["skin"], ec="k", lw=1.6, zorder=3))
    if p["style"] == "spiky":
        for k in range(5):
            xx = x - 0.7 * s + 0.35 * s * k
            ax.add_patch(plt.Polygon([[xx - 0.18 * s, y + 0.75 * s], [xx, y + 1.3 * s], [xx + 0.18 * s, y + 0.75 * s]],
                                     fc=p["hair"], ec="k", lw=1, zorder=4))
    elif p["style"] == "bob":
        ax.add_patch(Ellipse((x, y + 0.55 * s), 1.75 * s, 1.1 * s, angle=tilt, fc=p["hair"], ec="k", lw=1, zorder=2))
        ax.add_patch(Ellipse((x, y + 0.75 * s), 1.4 * s, 0.6 * s, angle=tilt, fc=p["hair"], ec="none", zorder=4))
    elif p["style"] == "flat":
        ax.add_patch(plt.Rectangle((x - 0.75 * s, y + 0.62 * s), 1.5 * s, 0.32 * s, angle=tilt, fc=p["hair"], ec="k",
                                   lw=1, zorder=4))
    dx = 0.32 * s
    for sgn in (-1, 1):
        ax.add_patch(plt.Circle((x + sgn * dx, y + 0.18 * s), 0.09 * s, fc="k", zorder=5))
    if variant % 3 == 1:
        ax.add_patch(Arc((x, y - 0.35 * s), 0.7 * s, 0.4 * s, theta1=200, theta2=340, lw=1.8, zorder=5))
    elif variant % 3 == 2:
        ax.plot([x - 0.25 * s, x + 0.25 * s], [y - 0.42 * s, y - 0.42 * s], color="k", lw=1.8, zorder=5)
    else:
        ax.add_patch(Arc((x, y - 0.3 * s), 0.6 * s, 0.35 * s, theta1=200, theta2=340, lw=1.8, zorder=5))


def fig_one_shot():
    with sketchy():
        fig, ax = plt.subplots(figsize=(12, 4.0))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 16), (-0.4, 4.8))
        ax.text(3.6, 4.35, "database: one photo per person", ha="center", fontsize=14, color=C["navy"])
        for k in range(4):
            draw_face(ax, 0.9 + 1.9 * k, 2.4, 0.7, k, 0)
            ax.text(0.9 + 1.9 * k, 0.7, f"person {k + 1}", ha="center", fontsize=12)
        ax.plot([8.3, 8.3], [0.2, 4.5], color=C["light"], lw=2)
        draw_face(ax, 10.0, 2.4, 0.7, 1, 1)
        ax.text(10.0, 0.7, "new photo", ha="center", fontsize=12)
        arrow(ax, (11.0, 2.4), (12.2, 2.4))
        ax.text(14.1, 2.6, "same as person 2", ha="center", fontsize=15, color=C["green"])
        ax.text(14.1, 1.9, "(or: \"not recognised\")", ha="center", fontsize=12, color=C["grey"])
        save(fig, "one_shot")


def fig_siamese():
    with sketchy():
        fig, ax = plt.subplots(figsize=(12, 4.6))
        fig.subplots_adjust(0, 0, 1, 1)
        blank(ax, (0, 16), (-0.6, 5.6))
        rng = np.random.default_rng(1)
        for row, (yc, person, var, lab) in enumerate([(4.0, 1, 0, "1"), (1.2, 1, 1, "2")]):
            draw_face(ax, 1.0, yc, 0.6, person, var)
            ax.text(1.0, yc - 1.05, f"$\\mathbf{{x}}^{{({lab})}}$", ha="center", fontsize=14)
            arrow(ax, (1.7, yc), (2.7, yc))
            box(ax, 5.0, yc, "CNN (same weights)", 4.4, 1.0, fc="#E3F2FD", ec=C["blue"], fs=14)
            arrow(ax, (7.25, yc), (8.2, yc))
            v = rng.uniform(0.2, 1.0, 8) if row == 0 else np.clip(v + rng.normal(0, 0.08, 8), 0.1, 1)
            for k in range(8):
                ax.add_patch(plt.Rectangle((8.4 + 0.42 * k, yc - 0.3), 0.42, 0.6, fc=plt.get_cmap("viridis")(v[k]),
                                           ec="k", lw=0.8))
            ax.text(10.1, yc + 0.55, f"encoding $\\mathbf{{f}}(\\mathbf{{x}}^{{({lab})}}) \\in \\mathbb{{R}}^{{128}}$",
                    ha="center", fontsize=13)
        arrow(ax, (12.0, 4.0), (13.2, 2.9), rad=-0.2)
        arrow(ax, (12.0, 1.2), (13.2, 2.3), rad=0.2)
        box(ax, 14.5, 2.6, "$d = \\|\\mathbf{f}(\\mathbf{x}^{(1)}) - \\mathbf{f}(\\mathbf{x}^{(2)})\\|^2$", 2.9, 1.0,
            fc="#FCE4EC", ec=C["red"], fs=12)
        ax.text(14.5, 1.6, "small: same person", ha="center", fontsize=12, color=C["green"])
        save(fig, "siamese")


def fig_triplet():
    with sketchy():
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.88, wspace=0.08)
        for ax, (P, N, t) in zip(axs, [((3.6, 3.4), (1.9, 1.0), "before training: P far, N close"),
                                        ((1.6, 2.4), (4.4, 0.6), "after training: margin $\\alpha$ between them")]):
            blank(ax, (-0.2, 5.6), (-0.6, 4.6))
            A = (1.0, 1.5)
            for (pt, person, var, lab, col) in [(A, 1, 0, "A", C["navy"]), (P, 1, 1, "P", C["green"]),
                                                 (N, 2, 2, "N", C["red"])]:
                draw_face(ax, *pt, 0.38, person, var)
                ax.text(pt[0] + 0.45, pt[1] + 0.45, lab, fontsize=16, color=col)
            ax.plot([A[0], P[0]], [A[1], P[1]], color=C["green"], lw=2, ls="--", zorder=1)
            ax.plot([A[0], N[0]], [A[1], N[1]], color=C["red"], lw=2, ls="--", zorder=1)
            ax.set_title(t, fontsize=14)
        fig.text(0.5, 0.02, "points: encodings $\\mathbf{f}(\\cdot)$ of anchor A, positive P (same person), "
                            "negative N (different person)", ha="center", fontsize=12, color=C["grey"])
        save(fig, "triplet")


# --------------------------------------------------------------------------- neural style transfer
def _style_mosaic(n=512, seed=0):
    """Style image 1: a stained-glass mosaic (Voronoi tiles, dark lead lines), generated here."""
    from scipy.ndimage import binary_dilation
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, n, size=(260, 2))
    gy, gx = np.mgrid[0:n, 0:n]
    _, lab = cKDTree(pts).query(np.c_[gx.ravel(), gy.ravel()])
    lab = lab.reshape(n, n)
    palette = np.array([[214, 40, 40], [247, 127, 0], [252, 191, 73], [0, 109, 119], [131, 197, 190],
                        [42, 157, 143], [38, 70, 83], [106, 76, 147], [233, 196, 106], [29, 53, 87]]) / 255
    img = palette[rng.integers(0, len(palette), len(pts))][lab]
    edge = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
    img[binary_dilation(edge, iterations=2)] = 0.08
    return img


def _style_swirl(n=512, seed=1):
    """Style image 2: swirling brush-stroke bands in blue and yellow, generated here."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    r = np.hypot(x - 0.5, y - 0.45)
    th = np.arctan2(y - 0.45, x - 0.5)
    noise = gaussian_filter(rng.normal(size=(n, n)), 3) * 6
    field = np.sin(16 * r + 3 * th + noise) + 0.5 * np.sin(9 * x + 5 * np.sin(6 * y))
    t = ((np.tanh(1.5 * field) + 1) / 2)[..., None]
    blue, light, yellow = np.array([0.08, 0.2, 0.5]), np.array([0.4, 0.62, 0.86]), np.array([0.98, 0.82, 0.25])
    img = np.where(t < 0.5, blue + (light - blue) * 2 * t, light + (yellow - light) * (2 * t - 1))
    return np.clip(img, 0, 1)


def _vgg_features():
    """VGG-19 convolutional layers pretrained on ImageNet (downloaded once by torchvision, about 550 MB)."""
    from torchvision import models

    return models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()


def _to_tensor(img, max_side):
    """(H, W, 3) array in [0, 1] -> (1, 3, h, w) tensor whose longer side is max_side."""
    import torch
    import torch.nn.functional as F

    t = torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1)[None]
    h, w = t.shape[2:]
    f = max_side / max(h, w)
    return F.interpolate(t, size=(round(h * f), round(w * f)), mode="bilinear", align_corners=False)


def _style_transfer(content, style, features, steps=300, snapshots=(), style_weight=1e6, content_weight=1.0):
    """Gatys et al. (2015), as in the PyTorch tutorial: optimise the pixels of G with L-BFGS.

    Content: activations of conv_4; style: Gram matrices of conv_1, ..., conv_5 (VGG-19 numbering by order).
    Returns the generated image and a dict {iteration: image} of snapshots, all as (H, W, 3) arrays.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    content = content.to(device)
    style = F.interpolate(style, size=content.shape[2:], mode="bilinear", align_corners=False).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    content_layers = {"conv_4"}
    style_layers = {"conv_1", "conv_2", "conv_3", "conv_4", "conv_5"}
    model, i = nn.Sequential(), 0
    for layer in features.children():                     # copy VGG up to conv_5, with tagged layer names
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name, layer = f"relu_{i}", nn.ReLU(inplace=False)
        else:
            name = f"pool_{i}"
        model.add_module(name, layer)
        if name == "conv_5":
            break
    model = model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)

    def activations(x):
        out, h = {}, (x - mean) / std
        for name, layer in model.named_children():
            h = layer(h)
            if name in content_layers or name in style_layers:
                out[name] = h
        return out

    def gram(a):
        b, c, h, w = a.shape
        f = a.view(b * c, h * w)
        return f @ f.T / (b * c * h * w)

    with torch.no_grad():
        target_c = {k: v for k, v in activations(content).items() if k in content_layers}
        target_s = {k: gram(v) for k, v in activations(style).items() if k in style_layers}
    G = content.clone().requires_grad_(True)              # start from the content image
    opt = torch.optim.LBFGS([G])
    count, pics = [0], {}
    to_np = lambda t: t.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()   # noqa: E731
    while count[0] < steps:
        def closure():
            with torch.no_grad():
                G.clamp_(0, 1)
            opt.zero_grad()
            act = activations(G)
            j_content = sum(F.mse_loss(act[k], target_c[k]) for k in content_layers)
            j_style = sum(F.mse_loss(gram(act[k]), target_s[k]) for k in style_layers)
            loss = content_weight * j_content + style_weight * j_style
            loss.backward()
            count[0] += 1
            if count[0] in snapshots:
                pics[count[0]] = to_np(G)
            return loss
        opt.step(closure)
    return to_np(G), pics


def _image_row(name, panels, height=3.4):
    with sketchy():
        fig, axs = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), height))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86, wspace=0.05)
        for ax, (img, title) in zip(np.atleast_1d(axs), panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])
        save(fig, name)


FORCE = False   # set by "--force" on the command line: redo neural style transfer even if its figures exist


def fig_style_transfer(steps=300):
    """Two style-transfer examples and the progress of the optimisation (slow: several minutes on a CPU).

    The optimisation only runs for figures that do not exist yet; use --force to compute them again.
    """
    need1 = FORCE or not all((OUT / f"{n}.png").exists() for n in ("nst_example_1", "nst_iterations"))
    need2 = FORCE or not (OUT / "nst_example_2.png").exists()
    if not (need1 or need2):
        print("    style-transfer figures exist already; skipped (use --force to compute them again)")
        return
    from PIL import Image

    photo = ROOT / "styles" / "backgrounds" / "source" / "title-photo.jpg"      # the University cloisters
    cloister = (np.asarray(Image.open(photo).convert("RGB")) / 255.0 if photo.exists()
                else load_sample_image("china.jpg") / 255.0)
    flower = load_sample_image("flower.jpg") / 255.0
    swirl, mosaic = _style_swirl(), _style_mosaic()
    import torch

    side = 384 if torch.cuda.is_available() else 256       # image size: smaller on a CPU to keep it fast
    features = _vgg_features()
    if need1:
        marks = tuple(sorted({k for k in (10, 50, 100, steps) if k <= steps}))
        g1, snaps = _style_transfer(_to_tensor(flower, side), _to_tensor(swirl, side), features, steps, marks)
        _image_row("nst_example_1", [(flower, "content image $C$"), (swirl, "style image $S$"),
                                     (g1, "generated image $G$")])
        _image_row("nst_iterations", [(snaps[k], f"after {k} iterations") for k in marks], height=2.9)
    else:
        print("    nst_example_1, nst_iterations exist already; skipped")
    if need2:
        g2, _ = _style_transfer(_to_tensor(cloister, side), _to_tensor(mosaic, side), features, steps)
        _image_row("nst_example_2", [(cloister, "content: the University cloisters"), (mosaic, "style: a mosaic"),
                                     (g2, "generated")])
    else:
        print("    nst_example_2 exists already; skipped")


FIGURES = {
    "cv_tasks": fig_cv_tasks, "sobel": fig_sobel, "conv_step": fig_conv_steps, "edge_example": fig_edge_example,
    "filters": fig_filters, "learned": fig_learned_filters_and_cnn_vs_mlp, "padding": fig_padding,
    "stride_step": fig_stride_steps, "volume": fig_volume, "pooling": fig_pooling, "architectures": fig_architectures,
    "transfer": fig_transfer, "augment": fig_augment, "one_shot": fig_one_shot, "siamese": fig_siamese,
    "triplet": fig_triplet,
    "nst": fig_style_transfer,                 # neural style transfer: slow, see the top of this file
}

if __name__ == "__main__":
    FORCE = "--force" in sys.argv[1:]
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    print(f"Font: {FONTS[0]}  ->  {OUT}")
    for key, fn in FIGURES.items():
        if not wanted or any(w in key for w in wanted):
            print(key)
            fn()
