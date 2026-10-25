"""
Generate the static illustrations used in the PyTorch notebooks.

    python tools/make_figures.py      # writes PNG files into ./figures

Only matplotlib and numpy are needed. Every figure is drawn from scratch,
so feel free to tweak colours, labels or sizes and re-run the script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# A small, consistent palette (University of Glasgow blues + accents)
BLUE = "#003865"      # dark blue
LIGHT = "#C6DBEF"     # pale blue
MID = "#5B8DB8"       # mid blue
ORANGE = "#E07B39"    # accent for "backward"
GREEN = "#3A8E5C"
GREY = "#6E6E6E"
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "mathtext.fontset": "cm",
})


def box(ax, xy, w, h, text, fc=LIGHT, ec=BLUE, fs=11, tc="black", lw=1.5,
        style="round,pad=0.02,rounding_size=0.08", **kw):
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                           fc=fc, ec=ec, lw=lw, **kw)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc)
    return patch


def arrow(ax, p, q, color=BLUE, lw=1.6, style="-|>", rad=0.0, ls="-", ms=14):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color,
                        lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------
# 1. Tensors of different ranks (Notebook 1)
# --------------------------------------------------------------------------
def tensor_ranks():
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.3),
                            gridspec_kw=dict(width_ratios=[0.6, 0.8, 1.1, 1.4]))
    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")

    def cell(ax, x, y, val, fc=LIGHT, size=1.0):
        ax.add_patch(Rectangle((x, y), size, size, fc=fc, ec=BLUE, lw=1.2))
        if val is not None:
            ax.text(x + size / 2, y + size / 2, val, ha="center", va="center",
                    fontsize=10)

    # scalar
    ax = axs[0]
    cell(ax, 0, 1.5, "7")
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-1.6, 4.0)
    ax.set_title("scalar\n0-D, shape ()", fontsize=11)

    # vector
    ax = axs[1]
    for i in range(4):
        cell(ax, 0, 3 - i, str(i + 1))
    ax.set_xlim(-1, 2); ax.set_ylim(-0.6, 4.6)
    ax.set_title("vector\n1-D, shape (4,)", fontsize=11)

    # matrix
    ax = axs[2]
    vals = np.arange(1, 13).reshape(4, 3)
    for i in range(4):
        for j in range(3):
            cell(ax, j, 3 - i, str(vals[i, j]))
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.6, 4.6)
    ax.set_title("matrix\n2-D, shape (4, 3)", fontsize=11)

    # 3-D tensor (stack of matrices)
    ax = axs[3]
    shades = ["#9ECAE1", "#C6DBEF", "#DEEBF7"]
    for k in reversed(range(3)):
        dx, dy = 0.55 * k, 0.55 * k
        for i in range(4):
            for j in range(3):
                cell(ax, j + dx, 3 - i + dy, None if k else "", fc=shades[k])
    ax.annotate("", xy=(4.9, 5.05), xytext=(3.2, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
    ax.text(4.3, 3.9, "dim 0", color=GREY, fontsize=9, rotation=42)
    ax.set_xlim(-0.5, 5.6); ax.set_ylim(-0.6, 5.6)
    ax.set_title("3-D tensor\nshape (3, 4, 3)", fontsize=11)

    save(fig, "tensor-ranks.png")


# --------------------------------------------------------------------------
# 2. Computational graph of out = sum(2 sin(x) + 1) (Notebook 2)
# --------------------------------------------------------------------------
def computational_graph():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(-1.0, 3.9); ax.axis("off")

    xs = [1.0, 3.5, 6.0, 8.5, 11.0]
    names = [r"$x$", r"$y$", r"$f$", r"$g$", r"$\mathrm{out}$"]
    descr = ["leaf tensor\nrequires_grad=True", "y = sin(x)", "f = 2 * y",
             "g = f + 1", "out = g.sum()"]
    grad_fn = ["AccumulateGrad\n(stores x.grad)", "SinBackward0", "MulBackward0",
               "AddBackward0", "SumBackward0"]
    local = [None, r"$\partial y/\partial x=\cos x$", r"$\partial f/\partial y=2$",
             r"$\partial g/\partial f=1$", r"$\partial\,\mathrm{out}/\partial g=1$"]
    y_fwd, y_bwd = 2.7, 0.75
    W = 1.95

    for i, x in enumerate(xs):
        fc = "#FFE8B3" if i == 0 else LIGHT
        box(ax, (x, y_fwd), W, 1.05, names[i] + "\n\n", fc=fc, fs=15)
        ax.text(x, y_fwd - 0.22, descr[i], ha="center", va="center",
                fontsize=8.3 if i == 0 else 9, color=BLUE, family="monospace")
        box(ax, (x, y_bwd), W, 0.75, grad_fn[i], fc="#FCE3D3", ec=ORANGE, fs=9)
        ax.plot([x, x], [y_fwd - 0.55, y_bwd + 0.4], ls=":", color=GREY, lw=1)

    for i in range(4):
        arrow(ax, (xs[i] + W / 2 + 0.05, y_fwd), (xs[i + 1] - W / 2 - 0.05, y_fwd),
              color=BLUE)
        arrow(ax, (xs[i + 1] - W / 2 - 0.05, y_bwd), (xs[i] + W / 2 + 0.05, y_bwd),
              color=ORANGE)
        ax.text((xs[i] + xs[i + 1]) / 2, y_bwd - 0.6, local[i + 1],
                ha="center", va="center", fontsize=10.5, color=ORANGE)

    ax.text(0.03, y_fwd + 0.85, "forward pass: autograd records every operation "
            "(the grad_fn of each result)", fontsize=10.5, color=BLUE, weight="bold")
    ax.text(0.03, -0.7,
            "backward pass: out.backward() walks the graph in reverse\n"
            "and multiplies the local derivatives (chain rule)",
            fontsize=10.5, color=ORANGE, weight="bold", va="center")
    ax.text(11.97, -0.7,
            r"$\dfrac{d\,\mathrm{out}}{dx}=1\cdot 1\cdot 2\cdot\cos x = 2\cos x$",
            ha="right", va="center", fontsize=13, color="black")
    save(fig, "computational-graph.png")


# --------------------------------------------------------------------------
# 3. The PyTorch training loop (Notebook 3)
# --------------------------------------------------------------------------
def training_loop():
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    ax.set_xlim(0, 10.5); ax.set_ylim(-0.1, 4.7); ax.axis("off")

    X1, X2, X3 = 1.6, 5.25, 8.9
    TOP, BOT = 3.35, 1.05
    W, H = 3.0, 1.05
    steps = [
        ((X1, TOP), "1. Reset gradients", "optimizer.zero_grad()"),
        ((X2, TOP), "2. Forward pass", "y_pred = model(X)"),
        ((X3, TOP), "3. Compute the loss", "loss = criterion(y_pred, y)"),
        ((X3, BOT), "4. Backward pass", "loss.backward()"),
        ((X2, BOT), "5. Update parameters", "optimizer.step()"),
    ]
    for (x, y), title, code in steps:
        box(ax, (x, y), W, H, "", fc=LIGHT)
        ax.text(x, y + 0.2, title, ha="center", va="center", fontsize=11,
                weight="bold", color=BLUE)
        ax.text(x, y - 0.2, code, ha="center", va="center", fontsize=9.3,
                family="monospace")

    arrow(ax, (X1 + W / 2 + 0.03, TOP), (X2 - W / 2 - 0.03, TOP))
    arrow(ax, (X2 + W / 2 + 0.03, TOP), (X3 - W / 2 - 0.03, TOP))
    arrow(ax, (X3, TOP - H / 2 - 0.03), (X3, BOT + H / 2 + 0.03))
    arrow(ax, (X3 - W / 2 - 0.03, BOT), (X2 + W / 2 + 0.03, BOT))
    arrow(ax, (X2 - W / 2 - 0.03, BOT), (X1, TOP - H / 2 - 0.03), rad=-0.35)
    ax.text(0.95, 1.05, "repeat for the next\nmini-batch / epoch", ha="center",
            fontsize=9.5, color=GREY, style="italic", va="center")
    ax.text(X3 - 0.15, 2.2, "computes p.grad for\nevery parameter p",
            fontsize=8.5, color=GREY, va="center", ha="right")
    ax.text(X1, TOP + 0.75, "otherwise new gradients are\nadded to the old ones",
            fontsize=8.5, color=GREY, ha="center", va="center")
    ax.text(X2, 0.05, "p  <-  p - lr * p.grad   (plain gradient descent; "
            "Adam & co. use a smarter rule)",
            ha="center", fontsize=8.5, color=GREY, family="monospace")
    ax.text(X2 + 1.8, 4.45, "model: an nn.Module   criterion: a loss from torch.nn"
            "   optimizer: from torch.optim",
            ha="center", fontsize=9, color=BLUE, family="monospace")
    save(fig, "training-loop.png")


# --------------------------------------------------------------------------
# 4. Binary logistic regression as a tiny network (Notebook 4)
# --------------------------------------------------------------------------
def logistic_regression():
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 9.5); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_aspect("equal")

    ys = [3.5, 2.6, 1.7, 0.6]
    labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_n$"]
    for y, lab in zip(ys, labels):
        ax.add_patch(plt.Circle((1.0, y), 0.33, fc="#FFE8B3", ec=BLUE, lw=1.4))
        ax.text(1.0, y, lab, ha="center", va="center", fontsize=13)
    ax.text(1.0, 1.15, r"$\vdots$", ha="center", va="center", fontsize=14)

    zx, zy = 4.4, 2.05
    box(ax, (zx, zy), 2.2, 1.0, r"$z=\mathbf{w}\cdot\mathbf{x}+b$", fs=13)
    for y, w in zip(ys, [r"$w_1$", r"$w_2$", r"$w_3$", r"$w_n$"]):
        end = (zx - 1.12, zy + 0.25 * (y - zy) / 1.5)
        arrow(ax, (1.35, y), end, lw=1.2)
        mx, my = 1.35 + 0.35 * (end[0] - 1.35), y + 0.35 * (end[1] - y)
        ax.text(mx, my + 0.12, w, fontsize=11, color=BLUE, ha="center")
    box(ax, (7.0, zy), 1.6, 1.0, r"$p=\sigma(z)$", fc="#D9F0E0", ec=GREEN, fs=13)
    arrow(ax, (zx + 1.12, zy), (6.18, zy))
    arrow(ax, (7.82, zy), (8.65, zy))
    ax.text(8.75, zy, r"$p$", fontsize=14, va="center")
    ax.text(zx, 0.85, "linear layer: nn.Linear(n, 1)", ha="center", fontsize=9,
            family="monospace", color=BLUE)
    ax.text(7.0, 0.85, "torch.sigmoid", ha="center", fontsize=9,
            family="monospace", color=GREEN)
    ax.text(7.0, 3.55, r"$p = P(y=1\mid\mathbf{x})$" + "\n"
            r"$\hat{y}=1$ if $p\geq 0.5$, else $\hat{y}=0$",
            ha="center", fontsize=11)
    save(fig, "logistic-regression.png")


# --------------------------------------------------------------------------
# 5. Multinomial (softmax) logistic regression for MNIST (Notebook 4)
# --------------------------------------------------------------------------
def softmax_regression():
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_xlim(0, 10.5); ax.set_ylim(-0.3, 4.6); ax.axis("off")

    # a fake 28x28 "image" -> 784 inputs
    rng = np.random.default_rng(3)
    img = np.zeros((28, 28))
    yy, xx = np.mgrid[0:28, 0:28]
    img += np.exp(-(((xx - 14) / 4.5) ** 2 + ((yy - 14) / 7.5) ** 2 - 1) ** 2 * 3)
    img = np.clip(img + 0.05 * rng.random((28, 28)), 0, 1)
    ax.imshow(img, cmap="gray", extent=(0.2, 1.8, 1.4, 3.0))
    ax.text(1.0, 1.15, "image\n(1, 28, 28)", ha="center", va="top", fontsize=9)
    arrow(ax, (1.95, 2.2), (2.75, 2.2))
    ax.text(2.35, 2.42, "flatten", ha="center", fontsize=8.5, color=GREY)

    # inputs column
    for k, y in enumerate([3.7, 3.1, 2.5, 1.0]):
        ax.add_patch(plt.Circle((3.2, y), 0.24, fc="#FFE8B3", ec=BLUE, lw=1.2))
    ax.text(3.2, 1.75, r"$\vdots$", ha="center", fontsize=13)
    ax.text(3.2, 0.45, "784 inputs", ha="center", fontsize=9)

    # logits column
    zs = [3.9, 3.2, 2.5, 0.9]
    zl = [r"$z_0$", r"$z_1$", r"$z_2$", r"$z_9$"]
    for y, lab in zip(zs, zl):
        box(ax, (5.6, y), 0.8, 0.5, lab, fs=12)
    ax.text(5.6, 1.7, r"$\vdots$", ha="center", fontsize=13)
    for yi in [3.7, 3.1, 2.5, 1.0]:
        for yo in zs:
            ax.plot([3.45, 5.18], [yi, yo], color=MID, lw=0.6, alpha=0.7)
    ax.text(5.6, 0.05, "logits\nnn.Linear(784, 10)", ha="center", fontsize=9,
            family="monospace", color=BLUE)

    # softmax
    box(ax, (7.35, 2.4), 1.25, 3.4, "softmax", fc="#D9F0E0", ec=GREEN, fs=11)
    for y in zs:
        arrow(ax, (6.02, y), (6.7, y), lw=1.0, ms=10)
    pl = [r"$p_0$", r"$p_1$", r"$p_2$", r"$p_9$"]
    for y, lab in zip(zs, pl):
        arrow(ax, (8.0, y), (8.6, y), lw=1.0, ms=10)
        ax.text(8.7, y, lab, va="center", fontsize=12)
    ax.text(8.75, 1.7, r"$\vdots$", ha="center", fontsize=13)
    ax.text(9.55, 2.45, r"$p_k=\dfrac{e^{z_k}}{\sum_{j} e^{z_j}}$",
            fontsize=12, va="center")
    ax.text(9.5, 0.45, r"$\hat{y}=\arg\max_k\, p_k$", fontsize=11, ha="center")
    ax.text(7.35, 4.35, "nn.CrossEntropyLoss takes the logits and "
            "applies (log-)softmax internally", ha="center", fontsize=9,
            color=GREEN)
    save(fig, "softmax-regression.png")


# --------------------------------------------------------------------------
# 6. JVP (columns) versus VJP (rows) of the Jacobian (Notebook 5)
# --------------------------------------------------------------------------
def jvp_vjp():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    rows, cols = 3, 4
    for ax, mode in zip(axs, ["jvp", "vjp"]):
        ax.set_xlim(-2.4, cols + 2.0); ax.set_ylim(-2.0, rows + 1.0)
        ax.set_aspect("equal"); ax.axis("off")
        hl_fc = "#D9F0E0" if mode == "jvp" else "#FCE3D3"
        hl_ec = GREEN if mode == "jvp" else ORANGE
        for i in range(rows):
            for j in range(cols):
                hl = (mode == "jvp" and j == 1) or (mode == "vjp" and i == 1)
                ax.add_patch(Rectangle((j, rows - 1 - i), 1, 1,
                                       fc=hl_fc if hl else LIGHT,
                                       ec=hl_ec if hl else BLUE, lw=1.4))
                ax.text(j + 0.5, rows - 1 - i + 0.5,
                        rf"$\frac{{\partial f_{i+1}}}{{\partial x_{j+1}}}$",
                        ha="center", va="center", fontsize=11)
        ax.text(cols / 2, rows + 0.35, r"$J\in\mathbb{R}^{3\times 4}$", ha="center",
                fontsize=11)
        s = 0.75
        if mode == "jvp":
            x0 = cols + 0.55
            for j in range(cols):
                y0 = rows - s - j * s
                ax.add_patch(Rectangle((x0, y0), s, s, fc=hl_fc if j == 1 else "white",
                                       ec=GREEN, lw=1.2))
                ax.text(x0 + s / 2, y0 + s / 2, "1" if j == 1 else "0",
                        ha="center", va="center", fontsize=11)
            ax.text(x0 + s / 2, -0.35, r"$\mathbf{v}=\mathbf{e}_2$", ha="center",
                    fontsize=11, color=GREEN)
            ax.text(cols / 2 + 0.3, -1.05,
                    r"JVP:  $J\mathbf{v}$ = column 2 = $\partial\mathbf{f}/\partial x_2$",
                    ha="center", fontsize=11.5, color=GREEN)
            ax.text(cols / 2 + 0.3, -1.7, "forward mode: one pass per input (n passes)",
                    ha="center", fontsize=9.5, color=GREY)
            ax.set_title("Jacobian-vector product (JVP)", fontsize=12,
                         color=GREEN, weight="bold")
        else:
            x0 = -1.5
            for i in range(rows):
                y0 = rows - 1 - i + (1 - s) / 2
                ax.add_patch(Rectangle((x0, y0), s, s, fc=hl_fc if i == 1 else "white",
                                       ec=ORANGE, lw=1.2))
                ax.text(x0 + s / 2, y0 + s / 2, "1" if i == 1 else "0",
                        ha="center", va="center", fontsize=11)
            ax.text(x0 + s / 2, -0.35, r"$\mathbf{u}=\mathbf{e}_2$", ha="center",
                    fontsize=11, color=ORANGE)
            ax.text(cols / 2 - 0.3, -1.05,
                    r"VJP:  $\mathbf{u}^{\top}J$ = row 2 = $\nabla f_2$",
                    ha="center", fontsize=11.5, color=ORANGE)
            ax.text(cols / 2 - 0.3, -1.7,
                    "reverse mode, backward(): one pass per output (l passes)",
                    ha="center", fontsize=9.5, color=GREY)
            ax.set_title("vector-Jacobian product (VJP)", fontsize=12,
                         color=ORANGE, weight="bold")
    save(fig, "jvp-vjp.png")


if __name__ == "__main__":
    tensor_ranks()
    computational_graph()
    training_loop()
    logistic_regression()
    softmax_regression()
    jvp_vjp()
