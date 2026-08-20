"""Frame strips from ACT's closed-loop ALOHA rollouts (the simulator's own render).

Companion to the Viser scene (05): that artifact abstracts the arms into EE splines,
this one shows the policy driving the actual MuJoCo bimanual ViperX rig, so a success
and a failure can be compared side by side.

Usage:
  python scripts/06_act_env_strip.py --run confirm_act/020000 --episodes 2 0
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path("outputs/eval")
OUT_DIR = Path("reports/m5/act_env")
N_COLS = 5


def read_frames(video: Path, n: int) -> list[np.ndarray]:
    """Decode n evenly-spaced frames as RGB arrays via a single ffmpeg pipe per frame."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True)
    total = int(probe.stdout.strip())
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{i})",
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True).stdout
        frames.append(np.frombuffer(raw, np.uint8).reshape(480, 640, 3))
    return frames


def episode_outcome(run: str, ep: int) -> tuple[bool, float]:
    m = json.loads((EVAL_DIR / run / "eval_info.json").read_text())["per_task"][0]["metrics"]
    return m["successes"][ep], m["max_rewards"][ep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="confirm_act/020000")
    ap.add_argument("--episodes", type=int, nargs="+", default=[2, 0])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vid_dir = EVAL_DIR / args.run / "videos" / "aloha_0"

    rows = len(args.episodes)
    fig, axes = plt.subplots(rows, N_COLS, figsize=(3.0 * N_COLS, 2.4 * rows))
    axes = np.atleast_2d(axes)

    for r, ep in enumerate(args.episodes):
        frames = read_frames(vid_dir / f"eval_episode_{ep}.mp4", N_COLS)
        ok, reward = episode_outcome(args.run, ep)
        for c, fr in enumerate(frames):
            ax = axes[r, c]
            # The MuJoCo top camera renders dark; lift it so the pegs read on a slide.
            ax.imshow(np.clip(fr.astype(np.float32) * 1.6, 0, 255).astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{c / (N_COLS - 1):.0%} through episode", fontsize=9)
        verdict = "SUCCESS" if ok else "FAILURE"
        axes[r, 0].set_ylabel(f"ep {ep} — {verdict}\nmax reward {reward:.0f}/4",
                              fontsize=10, color="green" if ok else "firebrick")

    fig.suptitle(f"ACT driving the ALOHA simulator ({args.run.replace('/', ' @ ')} steps)",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"rollout_{'_'.join(f'ep{e}' for e in args.episodes)}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
