"""Teaching aid: what a convolutional filter actually computes, on our own camera frames.

Four views:
  toy     — a 6x6 picture of two brightness values, so the whole computation can be
            checked by hand.
  layer2  — stacking: two edge maps from layer 1 combine into a corner detector.
  hand    — four hand-designed 3x3 kernels applied to a PushT frame, plus the literal
            multiply-and-add arithmetic at one pixel on a strong edge.
  learned — the 64 first-layer filters our trained policies actually learned, next to
            the ImageNet weights they started from, plus a few of their feature maps.

Usage:
  python scripts/07_cnn_filters.py toy
  python scripts/07_cnn_filters.py hand
  python scripts/07_cnn_filters.py learned
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEMO_DIR = Path("reports/m5/cnn_demo")
CACHE = Path.home() / ".cache/huggingface/hub"

KERNELS = {
    "vertical edge": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "horizontal edge": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "blur": [[1 / 9] * 3] * 3,
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
}


def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def convolve(img: np.ndarray, kernel: list[list[float]]) -> np.ndarray:
    x = torch.from_numpy(img)[None, None]
    k = torch.tensor(kernel, dtype=torch.float32)[None, None]
    return F.conv2d(x, k, padding=1)[0, 0].numpy()


def fmt_matrix(m: np.ndarray, width: int = 4) -> str:
    return "\n".join(" ".join(f"{v:>{width}.0f}" for v in row) for row in m)


def view_toy(_frame: Path) -> None:
    """Two brightness values, one filter, arithmetic small enough to verify mentally."""
    img = np.zeros((6, 6), dtype=np.float32)
    img[:, :3] = 10.0
    kern = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    scores = convolve(img, kern.tolist())[1:-1, 1:-1]  # valid positions only

    spots = {"A": (1, 1), "B": (1, 3)}

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.75], hspace=0.3, wspace=0.25)

    def annotate(ax, m, fmt="{:.0f}", color="tab:red", size=15):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                        color=color, fontsize=size, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, cmap="gray", vmin=0, vmax=10)
    annotate(ax, img)
    for name, (r, c) in spots.items():
        ax.add_patch(plt.Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False,
                                   color="tab:cyan", lw=3))
        ax.text(c, r - 1.8, name, ha="center", color="tab:cyan", fontsize=16,
                fontweight="bold")
    ax.set_title("the picture: 6x6 brightness numbers\nleft half bright (10), right half "
                 "dark (0)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(kern, cmap="RdBu", vmin=-1, vmax=1)
    annotate(ax, kern, fmt="{:+.0f}", color="black")
    ax.set_title("the filter: 9 weights\n'subtract the left, add the right'", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(scores, cmap="gray")
    annotate(ax, scores, color="tab:orange")
    ax.set_title("the scores, one per position\n0 = nothing here, 30 = edge here",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    for k, (name, (r, c)) in enumerate(spots.items()):
        patch = img[r - 1:r + 2, c - 1:c + 2]
        ax = fig.add_subplot(gs[1, k])
        ax.axis("off")
        lines = [f"spot {name}: the 9 numbers under the filter", ""]
        for i in range(3):
            row = "   ".join(f"{patch[i, j]:>2.0f} x {kern[i, j]:+.0f}" for j in range(3))
            lines.append(f"   {row}   =  {(patch[i] * kern[i]).sum():>+4.0f}")
        total = float((patch * kern).sum())
        verdict = "flat area, nothing found" if total == 0 else "edge found here"
        lines += ["", f"   total = {total:+.0f}    ({verdict})"]
        ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=12, family="monospace")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0.0, 1.0,
            "What just happened\n\n"
            "The filter is a scorecard that asks one\n"
            "question: 'is the left side brighter\n"
            "than the right side here?'\n\n"
            "Lay it on a spot, multiply, add.\n"
            "  same on both sides  ->  score 0\n"
            "  bright then dark    ->  big score\n\n"
            "Slide it everywhere and the scores\n"
            "form a map of where the answer is yes.",
            va="top", fontsize=12)

    fig.suptitle("A filter is a scorecard you slide over the picture", fontsize=14)
    out = DEMO_DIR / "toy_filter.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


def view_layer2(_frame: Path) -> None:
    """Layer 2 slides over layer 1's score maps: two edge maps combine into corners."""
    img = np.zeros((12, 12), dtype=np.float32)
    img[3:9, 3:9] = 10.0  # a bright square, so there are four corners to find

    # Thin filters (no smoothing along the edge direction) so the response does not
    # weaken at the corners, which is what muddies the classic Sobel version.
    vert = np.abs(convolve(img, [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]))
    horiz = np.abs(convolve(img, [[0, -1, 0], [0, 0, 0], [0, 1, 0]]))

    # A layer-2 filter sees both maps at once: 3x3 over each of the 2 input channels.
    stack = torch.from_numpy(np.stack([vert, horiz]))[None]
    summed = F.conv2d(stack, torch.ones(1, 2, 3, 3), padding=1)[0, 0].numpy()
    # Sum alone also fires along edges (60-70 there vs 80 at a corner). Bias + ReLU is
    # how a real CNN turns "either" into "both": the nonlinearity makes the AND possible.
    bias = 70.0
    corners = np.maximum(summed - bias, 0.0)

    panels = [
        (img, "the picture\na bright square", "gray"),
        (vert, "layer 1, filter A\nvertical edges", "magma"),
        (horiz, "layer 1, filter B\nhorizontal edges", "magma"),
        (summed, "layer 2, step 1\nadd up both maps nearby", "magma"),
        (corners, "layer 2, step 2\nsubtract 70, keep positives", "magma"),
    ]

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.2)
    for i, (m, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.imshow(m, cmap=cmap)
        peak = m.max()
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                if m[r, c] > 0.01:
                    ax.text(c, r, f"{m[r, c]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if m[r, c] < 0.6 * peak else "black")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0.0, 1.0,
            "Why this finds corners\n\n"
            "Along the flat top, only filter B fired.\n"
            "Along a side, only filter A fired.\n"
            "At a corner, BOTH fired.\n\n"
            "So adding the two maps scores 80 at a\n"
            "corner but only 60-70 along a straight\n"
            "edge. Subtract 70 and drop negatives:\n"
            "the edges are erased and the four\n"
            "corners are all that survive.\n\n"
            "Layer 2 never saw the picture. It only\n"
            "saw layer 1's scores -- and still ended\n"
            "up detecting something layer 1 could not.",
            va="top", fontsize=11)

    fig.suptitle("Layer 2 slides over layer 1's output, not over the picture",
                 fontsize=13)
    out = DEMO_DIR / "layer2_corners.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


def view_hand(frame: Path) -> None:
    img = gray(frame)
    responses = {name: convolve(img, k) for name, k in KERNELS.items()}

    # Anchor the arithmetic demo on the strongest vertical edge in the frame.
    vert = responses["vertical edge"]
    r, c = np.unravel_index(np.abs(vert[1:-1, 1:-1]).argmax(), vert[1:-1, 1:-1].shape)
    r, c = r + 1, c + 1
    patch = img[r - 1:r + 2, c - 1:c + 2]
    kern = np.array(KERNELS["vertical edge"])
    total = float((patch * kern).sum())

    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, cmap="gray")
    ax.add_patch(plt.Rectangle((c - 12, r - 12), 24, 24, fill=False, color="red", lw=2))
    ax.set_title(f"PushT camera frame\n(red box: pixel ({r}, {c}))", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{patch[i, j]:.0f}", ha="center", va="center",
                    color="red", fontsize=13, fontweight="bold")
    ax.set_title("the 3x3 pixels under the filter\n(brightness, 0-255)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(kern, cmap="RdBu", vmin=-2, vmax=2)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{kern[i, j]:+.0f}", ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_title("the filter (9 weights)\nblue = negative, red = positive", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    terms = " + ".join(f"({patch[i, j]:.0f}x{kern[i, j]:+.0f})"
                       for i in range(3) for j in range(3) if kern[i, j] != 0)
    ax.text(0.0, 0.95, "multiply each pair, add them up:", fontsize=11, va="top")
    ax.text(0.0, 0.80, terms.replace(" + ", "\n+ "), fontsize=9, va="top",
            family="monospace")
    ax.text(0.0, 0.16, f"= {total:+.0f}", fontsize=20, va="top", fontweight="bold",
            color="firebrick")
    ax.text(0.0, 0.05, "one output pixel. Slide the filter\nover the whole image to get "
                       "the maps below.", fontsize=9, va="top")

    for i, (name, resp) in enumerate(responses.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(np.abs(resp), cmap="magma")
        ax.set_title(f"{name}\n{fmt_matrix(np.array(KERNELS[name]) * (9 if name == 'blur' else 1))}",
                     fontsize=8, family="monospace")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("One convolutional filter, step by step (bottom row: brightness = "
                 "'this filter fired here')", fontsize=13)
    out = DEMO_DIR / "hand_filters.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


def load_conv1(repo: str) -> np.ndarray:
    """First conv layer of the ResNet18 image encoder inside a trained policy."""
    from safetensors.torch import load_file

    path = glob.glob(str(CACHE / f"models--nilakarthikesan--{repo}/snapshots/*/model.safetensors"))[0]
    sd = load_file(path)
    key = next(k for k in sd if k.endswith("conv1.weight") and sd[k].shape[-1] == 7)
    return sd[key].float().numpy()


def filter_grid(w: np.ndarray, ax, title: str, n: int = 32) -> None:
    """Show n filters as RGB tiles, each normalized to its own range."""
    tiles = []
    for f in w[:n]:
        f = f.transpose(1, 2, 0)
        f = (f - f.min()) / (np.ptp(f) + 1e-8)  # ndarray.ptp was removed in numpy 2.0
        tiles.append(np.pad(f, ((1, 1), (1, 1), (0, 0)), constant_values=1.0))
    rows = [np.concatenate(tiles[i:i + 8], axis=1) for i in range(0, n, 8)]
    ax.imshow(np.concatenate(rows, axis=0))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def view_learned(frame: Path) -> None:
    import torchvision

    imagenet = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    weights = {
        "ImageNet start point (resnet18)": imagenet.conv1.weight.detach().numpy(),
        "after Diffusion training on PushT": load_conv1("diffusion_pusht"),
        "after ACT training on ALOHA": load_conv1("act_aloha_insertion"),
    }

    rgb = np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1))[None]

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.25)

    for i, (name, w) in enumerate(weights.items()):
        filter_grid(w, fig.add_subplot(gs[0, i]), name)

    # What the trained PushT encoder's most active filters respond to in a real frame.
    w = torch.from_numpy(weights["after Diffusion training on PushT"])
    maps = F.conv2d(x, w, stride=2, padding=3)[0].numpy()
    order = np.argsort(-maps.reshape(len(maps), -1).std(axis=1))[:3]
    for j, idx in enumerate(order):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(np.abs(maps[idx]), cmap="magma")
        ax.set_title(f"filter #{idx} of the trained PushT encoder,\napplied to a real frame",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Learned filters: what training changes (top) and what they fire on "
                 "(bottom)", fontsize=13)
    out = DEMO_DIR / "learned_filters.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("view", choices=["toy", "layer2", "hand", "learned"])
    ap.add_argument("--frame", type=Path, default=DEMO_DIR / "frame_pusht.png")
    args = ap.parse_args()

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    views = {"toy": view_toy, "layer2": view_layer2, "hand": view_hand,
             "learned": view_learned}
    views[args.view](args.frame)


if __name__ == "__main__":
    main()
