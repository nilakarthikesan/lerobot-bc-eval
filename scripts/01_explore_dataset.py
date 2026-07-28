"""Dataset EDA (DESIGN.md stage 1).

Usage:
    python scripts/01_explore_dataset.py --repo-id lerobot/pusht
    python scripts/01_explore_dataset.py --repo-id lerobot/aloha_sim_insertion_human

For each dataset this prints episode/frame counts, fps, feature shapes, and the
episode-length distribution, and saves to reports/eda/<dataset>/:
  - stats.md                  summary table (pasteable into the writeup)
  - sample_frames.png         frames from one episode across time, per camera
  - episode_lengths.png       histogram of episode lengths
  - action_trajectory.png     action dims over time for one episode
                              (for pusht: also the 2D end-effector path)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def episode_bounds(ds: LeRobotDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (from, to) frame indices per episode.

    Note: v0.6.0 removed `episode_data_index`; boundaries now live in the
    `meta.episodes` Arrow table as dataset_from_index / dataset_to_index.
    """
    ep = ds.meta.episodes
    return np.asarray(ep["dataset_from_index"]), np.asarray(ep["dataset_to_index"])


def save_sample_frames(ds: LeRobotDataset, episode: int, out: Path, n: int = 5) -> None:
    starts, ends = episode_bounds(ds)
    frame_idxs = np.linspace(starts[episode], ends[episode] - 1, n).astype(int)
    cams = ds.meta.camera_keys
    fig, axes = plt.subplots(len(cams), n, figsize=(3 * n, 3 * len(cams)), squeeze=False)
    for col, fi in enumerate(frame_idxs):
        item = ds[int(fi)]
        for row, cam in enumerate(cams):
            img = item[cam]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()  # CHW float [0,1] -> HWC
            axes[row][col].imshow(np.clip(img, 0, 1))
            axes[row][col].set_axis_off()
            if row == 0:
                axes[row][col].set_title(f"t={int(fi - starts[episode])}", fontsize=9)
        axes[0][col].figure.tight_layout()
    for row, cam in enumerate(cams):
        axes[row][0].set_ylabel(cam, fontsize=8)
    fig.suptitle(f"{ds.repo_id} — episode {episode}", fontsize=11)
    fig.savefig(out / "sample_frames.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_episode_length_hist(lengths: np.ndarray, fps: float, repo_id: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=30, edgecolor="black", alpha=0.8)
    ax.axvline(lengths.mean(), color="red", linestyle="--", label=f"mean={lengths.mean():.0f}")
    ax.set_xlabel(f"episode length (frames @ {fps:g} fps)")
    ax.set_ylabel("count")
    ax.set_title(f"{repo_id} — episode length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "episode_lengths.png", dpi=120)
    plt.close(fig)


def save_action_trajectory(ds: LeRobotDataset, episode: int, out: Path) -> None:
    starts, ends = episode_bounds(ds)
    lo, hi = int(starts[episode]), int(ends[episode])
    hf_ds = ds.hf_dataset.select(range(lo, hi))
    actions = np.stack([np.asarray(a) for a in hf_ds["action"]])
    states = np.stack([np.asarray(s) for s in hf_ds["observation.state"]])

    is_2d_ee = actions.shape[1] == 2  # pusht: action *is* the 2D EE target
    ncols = 2 if is_2d_ee else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)

    ax = axes[0][0]
    for d in range(actions.shape[1]):
        ax.plot(actions[:, d], label=f"action[{d}]", linewidth=1)
    ax.set_xlabel("timestep")
    ax.set_ylabel("action value")
    ax.set_title(f"{ds.repo_id} — episode {episode} actions ({actions.shape[1]} dims)")
    if actions.shape[1] <= 14:
        ax.legend(fontsize=7, ncol=2)

    if is_2d_ee:
        ax2 = axes[0][1]
        ax2.plot(actions[:, 0], actions[:, 1], "-o", markersize=2, label="action (EE target)")
        ax2.plot(states[:, 0], states[:, 1], "-", alpha=0.6, label="state (EE position)")
        ax2.scatter(*actions[0], color="green", zorder=5, label="start")
        ax2.scatter(*actions[-1], color="red", zorder=5, label="end")
        ax2.set_title("2D end-effector path")
        ax2.set_aspect("equal")
        ax2.invert_yaxis()  # pusht workspace uses image-style coordinates
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "action_trajectory.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/pusht")
    parser.add_argument("--episode", type=int, default=0, help="episode used for frame/trajectory plots")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out = Path(args.out_dir or f"reports/eda/{args.repo_id.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset(args.repo_id)
    starts, ends = episode_bounds(ds)
    lengths = ends - starts
    fps = ds.meta.fps

    n_test = max(1, round(ds.num_episodes * 0.10))
    split_at = ds.num_episodes - n_test

    lines = [
        f"# EDA — `{args.repo_id}`",
        "",
        "| stat | value |",
        "|------|-------|",
        f"| episodes | {ds.num_episodes} |",
        f"| frames | {ds.num_frames} |",
        f"| fps | {fps} |",
        f"| camera keys | {', '.join(ds.meta.camera_keys)} |",
        f"| episode length | min {lengths.min()} / mean {lengths.mean():.1f} / max {lengths.max()} frames |",
        f"| proposed split | train episodes 0-{split_at - 1}, held-out {split_at}-{ds.num_episodes - 1} ({n_test} test episodes) |",
        "",
        "## Features",
        "",
        "| feature | dtype | shape |",
        "|---------|-------|-------|",
    ]
    for name, ft in ds.features.items():
        lines.append(f"| `{name}` | {ft.get('dtype')} | {ft.get('shape')} |")
    stats_md = "\n".join(lines) + "\n"
    (out / "stats.md").write_text(stats_md)
    print(stats_md)

    save_sample_frames(ds, args.episode, out)
    save_episode_length_hist(lengths, fps, args.repo_id, out)
    save_action_trajectory(ds, args.episode, out)
    print(f"saved: {out}/stats.md, sample_frames.png, episode_lengths.png, action_trajectory.png")


if __name__ == "__main__":
    main()
