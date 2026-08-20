"""M7 figures + metrics from the saved occlusion maps. Design: NOTES_SALIENCY.md.

Consumes outputs/saliency/*/step*_ep*.npz (saliency grids + the frames they were computed
on) and outputs/eval/screen_*/*/eval_info.json (closed-loop ground truth). Never runs
inference — the probe (08_saliency.py) already did.

Two label-free metrics per checkpoint:
  focus         = peak / mean of the saliency map (averaged over frames)
  motion_overlap= fraction of saliency mass inside the episode's motion region

Usage:
  python scripts/09_saliency_figures.py            # metrics table + F1/F2/F3
  python scripts/09_saliency_figures.py --metrics  # just print the table
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAL_DIR = Path("outputs/saliency")
EVAL_DIR = Path("outputs/eval")
REPORT_DIR = Path("reports/m7")

# Which checkpoints were probed, and the screen dir that holds their success rates.
RUNS = {
    "act_aloha": {"screen": "screen_act", "steps": ["020000", "040000", "060000", "080000", "100000"], "ep": 47},
    "diffusion_pusht": {"screen": "screen_diffusion", "steps": ["025000", "100000", "200000"], "ep": 190},
}


def success_rate(screen: str, step: str) -> float | None:
    f = EVAL_DIR / screen / step / "eval_info.json"
    if not f.exists():
        return None
    return float(json.load(open(f))["overall"]["pc_success"])


def load_run(policy: str, step: str, ep: int) -> dict:
    return dict(np.load(SAL_DIR / policy / f"step{step}_ep{ep:03d}.npz", allow_pickle=True))


def motion_mask_grid(frames_rgb: np.ndarray, gh: int, gw: int, thresh_pct: float = 60.0) -> np.ndarray:
    """Where the scene changes across the episode, pooled to the saliency grid.

    frames_rgb [F, H, W, 3] -> per-pixel temporal std -> average into gh x gw cells ->
    keep the top (100-thresh_pct)% most-moving cells. A label-free stand-in for
    'task-relevant region' (the pusher/T sweep, the arm's contact zone).
    """
    gray = frames_rgb.mean(axis=-1)  # [F, H, W]
    motion = gray.std(axis=0)  # [H, W]
    H, W = motion.shape
    ys = np.linspace(0, H, gh + 1).astype(int)
    xs = np.linspace(0, W, gw + 1).astype(int)
    cell = np.array([[motion[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].mean()
                      for c in range(gw)] for r in range(gh)])
    return cell >= np.percentile(cell, thresh_pct)


def metrics_for(sal: np.ndarray, frames_rgb: np.ndarray) -> tuple[float, float]:
    """sal [F, gh, gw]. Returns (focus, motion_overlap), averaged over frames."""
    gh, gw = sal.shape[1:]
    mask = motion_mask_grid(frames_rgb, gh, gw)
    focus, overlap = [], []
    for m in sal:
        focus.append(m.max() / m.mean())
        overlap.append((m * mask).sum() / m.sum())
    return float(np.mean(focus)), float(np.mean(overlap))


def collect() -> dict:
    """policy -> list of dicts with step, success, focus, overlap, and the raw run."""
    out = {}
    for policy, cfg in RUNS.items():
        rows = []
        for step in cfg["steps"]:
            run = load_run(policy, step, cfg["ep"])
            focus, overlap = metrics_for(run["saliency"], run["frames_rgb"])
            rows.append({
                "step": int(step), "success": success_rate(cfg["screen"], step),
                "focus": focus, "overlap": overlap, "run": run,
            })
        out[policy] = rows
    return out


def print_table(data: dict) -> None:
    for policy, rows in data.items():
        print(f"\n== {policy} ==")
        print(f"{'step':>7} {'success%':>9} {'focus':>7} {'motion_overlap':>15}")
        for r in rows:
            s = "n/a" if r["success"] is None else f"{r['success']:.0f}"
            print(f"{r['step']:>7} {s:>9} {r['focus']:>7.1f} {r['overlap']:>15.3f}")
        succ = np.array([r["success"] for r in rows], dtype=float)
        if np.isfinite(succ).sum() >= 3 and np.nanstd(succ) > 0:
            for key in ("focus", "overlap"):
                v = np.array([r[key] for r in rows])
                c = np.corrcoef(v, succ)[0, 1]
                print(f"   Pearson r(success, {key}) = {c:+.2f}")


# ---------- figures ----------

def overlay(ax, rgb: np.ndarray, sal: np.ndarray, title: str, mask=None) -> None:
    ax.imshow(rgb)
    ax.imshow(sal, cmap="magma", alpha=0.55, extent=[0, rgb.shape[1], rgb.shape[0], 0],
              interpolation="bilinear", aspect="auto")
    if mask is not None:
        ax.contour(np.linspace(0, rgb.shape[1], mask.shape[1]),
                   np.linspace(0, rgb.shape[0], mask.shape[0]), mask.astype(float),
                   levels=[0.5], colors="cyan", linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def fig_heatmaps(data: dict, policy: str, step_lo: str, step_hi: str, fname: str,
                 label_lo: str, label_hi: str, n_show: int = 3) -> None:
    rows = {str(r["step"]): r for r in data[policy]}
    lo, hi = rows[str(int(step_lo))], rows[str(int(step_hi))]
    F = lo["run"]["saliency"].shape[0]
    idxs = np.linspace(0, F - 1, n_show).astype(int)
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    for j, fi in enumerate(idxs):
        rgb = lo["run"]["frames_rgb"][fi]
        gh, gw = lo["run"]["saliency"].shape[1:]
        mask = motion_mask_grid(lo["run"]["frames_rgb"], gh, gw)
        overlay(axes[0, j], rgb, lo["run"]["saliency"][fi],
                f"{label_lo}  frame {lo['run']['frames_ts'][fi]}", mask)
        overlay(axes[1, j], hi["run"]["frames_rgb"][fi], hi["run"]["saliency"][fi],
                f"{label_hi}  frame {hi['run']['frames_ts'][fi]}", mask)
    axes[0, 0].set_ylabel(label_lo, fontsize=11)
    axes[1, 0].set_ylabel(label_hi, fontsize=11)
    fig.suptitle(f"Occlusion saliency (cyan = region that moves during the episode)\n"
                 f"{label_lo}: {lo['success']:.0f}% success   vs   {label_hi}: {hi['success']:.0f}% success",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = REPORT_DIR / fname
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_metric_vs_step(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles = {"act_aloha": "ACT (ALOHA insertion)", "diffusion_pusht": "Diffusion (PushT)"}
    for ax, (policy, rows) in zip(axes, data.items()):
        steps = [r["step"] / 1000 for r in rows]
        succ = [r["success"] for r in rows]
        focus = [r["focus"] for r in rows]
        overlap = [r["overlap"] for r in rows]
        ax.bar(steps, succ, width=6, color="0.85", label="closed-loop success %", zorder=1)
        ax.set_ylabel("closed-loop success %")
        ax.set_ylim(0, 100)
        ax.set_xlabel("training step (thousands)")
        ax.set_title(titles[policy])
        ax2 = ax.twinx()
        ax2.plot(steps, focus, "o-", color="firebrick", label="focus (peak/mean)")
        ax2.plot(steps, [o * 100 for o in overlap], "s--", color="steelblue",
                 label="motion overlap (x100)")
        ax2.set_ylabel("focus  /  motion overlap x100")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    fig.suptitle("Does the encoder's attention track trustworthiness? "
                 "(bars = ground-truth success, lines = laptop-speed probe)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = REPORT_DIR / "F2_metric_vs_step.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", action="store_true", help="print the table only, skip figures")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    print_table(data)
    if args.metrics:
        return

    fig_heatmaps(data, "act_aloha", "020000", "100000", "F1_act_20k_vs_100k.png",
                 "ACT 20K", "ACT 100K")
    fig_metric_vs_step(data)
    fig_heatmaps(data, "diffusion_pusht", "025000", "200000", "F3_diffusion_25k_vs_200k.png",
                 "Diffusion 25K", "Diffusion 200K")


if __name__ == "__main__":
    main()
