"""Stage 4 (M5): figures from M4's replay arrays. Design: NOTES_VISUALIZATION.md.

Consumes outputs/mock_deploy/*/ep*.npz + metrics.json + dataset video frames.
Never re-runs policy inference.

Usage:
  python scripts/04_visualize.py frame-test              # V1-V3 verification stills
  python scripts/04_visualize.py depth                   # F1
  python scripts/04_visualize.py pusht-frames [--episode 190]   # F2
  python scripts/04_visualize.py pusht-fans   [--episode 190]   # F3
  python scripts/04_visualize.py pusht-video  [--episode 190]   # F4
  python scripts/04_visualize.py aloha-joints [--episode 47]    # F5
  python scripts/04_visualize.py aloha-heatmap [--episode 47]   # F6
  python scripts/04_visualize.py all
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path("reports/m5")
MOCK_DIR = Path("outputs/mock_deploy")
# PushT: actions live in the 512x512 pymunk workspace, frames are 96x96 renders.
# No y-flip: verified frame-by-frame (NOTES_VISUALIZATION.md V1) — gym-pusht renders
# with the same y-down convention the action space uses.
PUSHT_SCALE = 96 / 512


def load_ep(policy: str, ep: int) -> dict:
    return dict(np.load(MOCK_DIR / policy / f"ep{ep:03d}.npz"))


def pusht_frames(ep: int, frame_idxs: list[int]) -> list[np.ndarray]:
    """Decode the episode's stored 96x96 frames at the given local indices."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/pusht", episodes=[ep])
    return [ds[i]["observation.image"].permute(1, 2, 0).numpy() for i in frame_idxs]


def to_px(a: np.ndarray) -> np.ndarray:
    return a * PUSHT_SCALE


def draw_traj(ax, traj_ws: np.ndarray, color: str, label: str | None = None, lw: float = 1.8):
    """Trajectory in workspace coords, alpha fading with prediction depth."""
    p = to_px(traj_ws)
    n = len(p)
    for i in range(n - 1):
        ax.plot(p[i : i + 2, 0], p[i : i + 2, 1], color=color, lw=lw,
                alpha=0.9 - 0.7 * i / max(n - 1, 1))
    ax.plot(p[0, 0], p[0, 1], "o", color=color, ms=4, label=label)


def overlay_state(ax, frame: np.ndarray, gt: np.ndarray, pred: np.ndarray | None,
                  pad: np.ndarray, title: str):
    ax.imshow(frame)
    draw_traj(ax, gt[~pad], "lime", "ground truth")
    if pred is not None:
        draw_traj(ax, pred, "red", "predicted")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 96)
    ax.set_ylim(96, 0)  # image convention: y down
    ax.axis("off")


def fig_frame_test(ep: int) -> None:
    """V1-V3: GT + prediction overlays at 3 spread states, one PNG to eyeball."""
    d = load_ep("diffusion_pusht", ep)
    picks = [2, len(d["ts"]) // 2, len(d["ts"]) - 8]
    frames = pusht_frames(ep, [int(d["ts"][i]) for i in picks])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, i, fr in zip(axes, picks, frames):
        overlay_state(ax, fr, d["gt"][i], d["pred"][i], d["pad"][i], f"ep {ep}  t={int(d['ts'][i])}")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("V1-V3 frame test: green must track the pusher; red must start beside it", fontsize=10)
    out = REPORT_DIR / f"frame_test_ep{ep:03d}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_depth() -> None:
    """F1: RMSE-by-depth curves, one panel per policy (units differ)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    specs = [
        ("diffusion_pusht", "Diffusion / PushT", "px (512-px workspace)", "tab:blue"),
        ("act_aloha", "ACT / ALOHA insertion", "rad (joint space)", "tab:orange"),
    ]
    for ax, (policy, title, unit, color) in zip(axes, specs):
        m = json.load(open(MOCK_DIR / policy / "metrics.json"))
        rmse = np.sqrt(m["mse_by_depth"])
        ax.plot(range(1, len(rmse) + 1), rmse, color=color, lw=2)
        ax.set_title(f"{title}\nRMSE {rmse[0]:.3g} → {rmse[-1]:.3g}", fontsize=10)
        ax.set_xlabel("prediction depth d (steps ahead)")
        ax.set_ylabel(f"RMSE(d), {unit}")
        ax.grid(alpha=0.3)
    fig.suptitle("Open-loop error compounds with prediction depth (held-out episodes)", fontsize=11)
    fig.tight_layout()
    out = REPORT_DIR / "depth_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_pusht_frames(ep: int, n_stills: int = 6) -> None:
    """F2: a row of stills across the episode with GT + prediction overlays."""
    d = load_ep("diffusion_pusht", ep)
    picks = np.linspace(0, len(d["ts"]) - 1, n_stills).astype(int)
    frames = pusht_frames(ep, [int(d["ts"][i]) for i in picks])
    fig, axes = plt.subplots(1, n_stills, figsize=(2.6 * n_stills, 3.1))
    for ax, i, fr in zip(axes, picks, frames):
        overlay_state(ax, fr, d["gt"][i], d["pred"][i], d["pad"][i], f"t={int(d['ts'][i])}")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"PushT episode {ep}: predicted 32-step chunk (red) vs demonstrator (green)", fontsize=11)
    fig.tight_layout()
    out = REPORT_DIR / f"pusht_ep{ep:03d}_frames.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_pusht_fans(max_states: int = 4) -> None:
    """F3: K=8 diffusion samples at the HIGHEST-spread probe states across all episodes.

    Finding (NOTES_VISUALIZATION.md §6 V4): the trained policy is near-deterministic
    given context — median across-sample std is ~1.6 px of the 512-px workspace. So
    instead of arbitrary states (whose fans collapse to a bundle), we rank every probe
    state by max-over-depth spread and show the top ones, spread quantified in-title.
    """
    ranked = []
    for f in sorted((MOCK_DIR / "diffusion_pusht").glob("ep*.npz")):
        d = np.load(f)
        std = d["multi_pred"].std(axis=1).mean(axis=-1)  # [Tm, H] across-sample spread
        for i, t in enumerate(d["multi_ts"]):
            ranked.append((float(std[i].max()), int(f.stem[2:]), i, int(t)))
    ranked.sort(reverse=True)
    picks = ranked[:max_states]

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, max_states, figsize=(3.0 * max_states, 3.4))
    for ax, (spread, ep, i, t) in zip(np.atleast_1d(axes), picks):
        d = load_ep("diffusion_pusht", ep)
        fr = pusht_frames(ep, [t])[0]
        q = np.nonzero(d["ts"] == t)[0][0]  # queried-state index matching this probe
        ax.imshow(fr)
        for k in range(d["multi_pred"].shape[1]):
            p = to_px(d["multi_pred"][i, k])
            ax.plot(p[:, 0], p[:, 1], color=cmap(k), lw=1.3, alpha=0.6)
        g = to_px(d["gt"][q][~d["pad"][q]])
        ax.plot(g[:, 0], g[:, 1], color="lime", lw=2.2, label="ground truth")
        ax.set_title(f"ep {ep}  t={t}  spread {spread:.1f}px", fontsize=9)
        ax.set_xlim(0, 96)
        ax.set_ylim(96, 0)
        ax.axis("off")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(
        "PushT: 8 diffusion samples at the highest-spread held-out states\n"
        "(spread = max-over-depth std across samples, in 512-px workspace units)",
        fontsize=10,
    )
    fig.tight_layout()
    out = REPORT_DIR / "pusht_fans_top_spread.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_pusht_video(ep: int, fps: int = 5) -> None:
    """F4: F2-style overlay animated over every queried state -> mp4 via ffmpeg."""
    d = load_ep("diffusion_pusht", ep)
    frames = pusht_frames(ep, [int(t) for t in d["ts"]])
    tmp = Path(tempfile.mkdtemp(prefix="m5_vid_"))
    for j, (i, fr) in enumerate(zip(range(len(d["ts"])), frames)):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        overlay_state(ax, fr, d["gt"][i], d["pred"][i], d["pad"][i], f"episode {ep}  t={int(d['ts'][i])}")
        if j == 0:
            ax.legend(loc="lower left", fontsize=7)
        fig.tight_layout()
        fig.savefig(tmp / f"{j:04d}.png", dpi=110)
        plt.close(fig)
    out = REPORT_DIR / f"pusht_ep{ep:03d}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(tmp / "%04d.png"), "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)],
        check=True,
    )
    shutil.rmtree(tmp)
    print(f"wrote {out} ({len(frames)} frames)")


def fig_aloha_joints(ep: int, n_anchors: int = 5) -> None:
    """F5: 14 per-joint panels; predicted 100-step chunks branch off the GT trace."""
    d = load_ep("act_aloha", ep)
    gt_traj = d["gt"][:, 0, :]  # depth-0 action at every state = the episode's GT actions
    anchors = np.linspace(0, len(d["ts"]) - 110, n_anchors).astype(int)
    cmap = plt.get_cmap("viridis")
    names = [f"{side}.{j}" for side in ("L", "R") for j in
             ("waist", "shoulder", "elbow", "forearm", "wrist_a", "wrist_r", "grip")]
    fig, axes = plt.subplots(7, 2, figsize=(11, 13), sharex=True)
    for j, ax in enumerate(axes.T.flat):
        ax.plot(d["ts"], gt_traj[:, j], color="0.3", lw=1.0, label="ground truth" if j == 0 else None)
        for a_i, a in enumerate(anchors):
            t0 = d["ts"][a]
            chunk = d["pred"][a, :, j]
            ax.plot(np.arange(t0, t0 + len(chunk)), chunk, color=cmap(a_i / max(n_anchors - 1, 1)),
                    lw=1.4, alpha=0.9)
        ax.set_ylabel(names[j], fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("episode frame (50 fps)")
    axes[-1, 1].set_xlabel("episode frame (50 fps)")
    fig.suptitle(
        f"ALOHA episode {ep}: ACT 100-step chunks (colored by anchor state) vs demonstrator (grey)",
        fontsize=12,
    )
    fig.tight_layout()
    out = REPORT_DIR / f"aloha_ep{ep:03d}_joints.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def fig_aloha_heatmap(ep: int) -> None:
    """F6: |pred - gt| by (state, depth), joints averaged — when is ACT hard to predict?"""
    d = load_ep("act_aloha", ep)
    err = np.abs(d["pred"] - d["gt"]).mean(axis=2)  # [T, 100]
    err[d["pad"]] = np.nan
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(err.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, len(d["ts"]), 1, err.shape[1]])
    ax.set_xlabel("episode frame (anchor state)")
    ax.set_ylabel("prediction depth d")
    ax.set_title(f"ALOHA episode {ep}: mean |pred − gt| (rad) by state × depth", fontsize=11)
    fig.colorbar(im, ax=ax, label="rad")
    fig.tight_layout()
    out = REPORT_DIR / f"aloha_ep{ep:03d}_error_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figure", choices=["frame-test", "depth", "pusht-frames", "pusht-fans",
                                       "pusht-video", "aloha-joints", "aloha-heatmap", "all"])
    ap.add_argument("--episode", type=int, default=None)
    args = ap.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p_ep = args.episode if args.episode is not None else 190
    a_ep = args.episode if args.episode is not None else 47

    if args.figure == "frame-test":
        fig_frame_test(p_ep)
    if args.figure in ("depth", "all"):
        fig_depth()
    if args.figure in ("pusht-frames", "all"):
        fig_pusht_frames(p_ep)
    if args.figure in ("pusht-fans", "all"):
        fig_pusht_fans()
    if args.figure in ("pusht-video", "all"):
        fig_pusht_video(p_ep)
    if args.figure in ("aloha-joints", "all"):
        fig_aloha_joints(a_ep)
    if args.figure in ("aloha-heatmap", "all"):
        fig_aloha_heatmap(a_ep)


if __name__ == "__main__":
    main()
