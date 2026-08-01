"""Mock deployment part (a): open-loop replay on held-out episodes.

Design: NOTES_DEPLOYMENT.md §2. For every state of every held-out episode, query the
trained policy for its FULL action chunk and record it next to what the human
demonstrator actually did. Never uses select_action() (queue trap, §2); alignment
chunk-step-d == GT-action-at-t+d verified against the modeling code.

Usage:
  python scripts/03_mock_deploy.py --policy diffusion_pusht
  python scripts/03_mock_deploy.py --policy act_aloha
  python scripts/03_mock_deploy.py --policy diffusion_pusht --episodes 190,191 --stride 5

Outputs (consumed by 04_visualize.py, never re-runs inference):
  outputs/mock_deploy/<policy>/ep<id>.npz   ts, pred [T,H,A], gt [T,H,A], pad [T,H],
                                            multi_ts, multi_pred [Tm,K,H,A]
  outputs/mock_deploy/<policy>/metrics.json mse_by_depth, per-episode stats, config echo
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

PRESETS = {
    "diffusion_pusht": {
        "policy_cls": DiffusionPolicy,
        "policy_repo": "nilakarthikesan/diffusion_pusht",
        "dataset_repo": "lerobot/pusht",
        "episodes": list(range(185, 206)),  # held out from training (DESIGN.md §2)
        "k_samples": 8,  # multimodality probe (NOTES_DEPLOYMENT.md §2 metrics)
    },
    "act_aloha": {
        "policy_cls": ACTPolicy,
        "policy_repo": "nilakarthikesan/act_aloha_insertion",
        "dataset_repo": "lerobot/aloha_sim_insertion_human",
        "episodes": list(range(45, 50)),
        "k_samples": 1,  # ACT is deterministic at inference (KL collapsed; ACT notes)
    },
}


def build_dataset(preset: dict, cfg, episodes: list[int]) -> LeRobotDataset:
    """Dataset windows per the checkpoint's I/O contract (NOTES_DEPLOYMENT.md §2 table)."""
    fps_probe = LeRobotDataset(preset["dataset_repo"], episodes=[episodes[0]])
    fps = fps_probe.meta.fps
    # Only window obs keys when the policy defines a window: requesting even a length-1
    # window adds a time dim (B, 1, ...) that ACT's forward rejects (issue D10).
    delta_timestamps = {}
    if cfg.observation_delta_indices is not None:
        deltas = [i / fps for i in cfg.observation_delta_indices]
        delta_timestamps = {key: deltas for key in cfg.input_features}
    # GT window = the executed chunk, deltas 0..n_action_steps-1 (alignment verified:
    # both policies return the chunk starting at the current frame).
    delta_timestamps["action"] = [i / fps for i in range(cfg.n_action_steps)]
    return LeRobotDataset(preset["dataset_repo"], episodes=episodes, delta_timestamps=delta_timestamps)


def episode_slices(ds: LeRobotDataset, episodes: list[int]) -> list[tuple[int, int, int]]:
    """(episode_id, from_idx, to_idx) per episode, in SUBSET-local coordinates.

    Gotcha (issue D9): with an episode subset, `meta.episodes` still holds the FULL
    episode table with GLOBAL frame indices, while the dataset itself only contains the
    subset's frames — so we filter to the selected episodes and rebuild local offsets
    from the episode lengths (frames are stored in ascending episode order).
    """
    ep = ds.meta.episodes
    lengths = {
        int(e): int(t) - int(f)
        for e, f, t in zip(ep["episode_index"], ep["dataset_from_index"], ep["dataset_to_index"])
    }
    out, offset = [], 0
    for e in sorted(episodes):
        out.append((e, offset, offset + lengths[e]))
        offset += lengths[e]
    assert offset == len(ds), f"episode lengths sum to {offset}, dataset has {len(ds)} frames"
    return out


def predict_batch(policy, pre, post, items: list[dict], obs_keys: list[str], device) -> np.ndarray:
    """Stack items -> preprocess -> chunk -> postprocess. Returns [B, H, A] numpy."""
    batch = {k: torch.stack([it[k] for it in items]) for k in obs_keys}
    proc = pre(batch)
    proc = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in proc.items()}
    with torch.no_grad():
        chunk = policy.predict_action_chunk(proc)
    return post(chunk).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, choices=sorted(PRESETS))
    ap.add_argument("--episodes", default=None, help="comma-separated override, e.g. 190,191")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="query every Nth state for metrics")
    ap.add_argument("--k-samples", type=int, default=None, help="override preset K for the probe")
    ap.add_argument("--multi-stride", type=int, default=10, help="probe every Nth queried state")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    preset = PRESETS[args.policy]
    episodes = [int(e) for e in args.episodes.split(",")] if args.episodes else preset["episodes"]
    k_samples = args.k_samples if args.k_samples is not None else preset["k_samples"]
    out_dir = Path(args.output_dir or f"outputs/mock_deploy/{args.policy}")
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = preset["policy_cls"].from_pretrained(preset["policy_repo"])
    policy.eval()
    device = next(policy.parameters()).device
    cfg = policy.config
    obs_keys = list(cfg.input_features)
    # Saved pipeline hard-codes the training device (cuda) -> override (issue D5).
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=preset["policy_repo"],
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"{args.policy}: chunk={cfg.n_action_steps}, obs={obs_keys}, device={device}")

    ds = build_dataset(preset, cfg, episodes)
    horizon = cfg.n_action_steps
    action_dim = ds[0]["action"].shape[-1]

    # Accumulators for MSE(d): mean over valid states of mean-over-dims sq. error at depth d.
    sq_sum = np.zeros(horizon)
    n_valid = np.zeros(horizon)
    per_episode = []
    t_start = time.time()

    for ep_id, ep_from, ep_to in episode_slices(ds, episodes):
        idxs = list(range(ep_from, ep_to, args.stride))
        items = [ds[i] for i in idxs]
        gt = np.stack([it["action"].numpy() for it in items])  # [T, H, A]
        pad = np.stack([it["action_is_pad"].numpy() for it in items])  # [T, H] True = padded

        pred = np.concatenate(
            [
                predict_batch(policy, pre, post, items[b : b + args.batch_size], obs_keys, device)
                for b in range(0, len(items), args.batch_size)
            ]
        )  # [T, H, A]

        # Multimodality probe: K fresh-noise samples at strided states (diffusion only).
        multi_local = list(range(0, len(items), args.multi_stride))
        if k_samples > 1:
            probe_items = [items[i] for i in multi_local]
            multi_pred = np.stack(
                [
                    predict_batch(policy, pre, post, probe_items, obs_keys, device)
                    for _ in range(k_samples)
                ],
                axis=1,
            )  # [Tm, K, H, A]
        else:
            multi_pred = pred[multi_local][:, None]  # [Tm, 1, H, A]

        err = ((pred - gt) ** 2).mean(axis=-1)  # [T, H] per-depth error, mean over dims
        valid = ~pad
        sq_sum += np.where(valid, err, 0.0).sum(axis=0)
        n_valid += valid.sum(axis=0)
        ep_mse = float(err[valid].mean())
        per_episode.append(
            {"episode": ep_id, "n_states": len(items), "mse": ep_mse, "rmse": float(np.sqrt(ep_mse))}
        )
        print(f"episode {ep_id}: {len(items)} states, MSE {ep_mse:.4f} ({time.time() - t_start:.0f}s)")

        ts = np.array(idxs) - ep_from  # frame index within the episode
        np.savez_compressed(
            out_dir / f"ep{ep_id:03d}.npz",
            ts=ts,
            pred=pred.astype(np.float32),
            gt=gt.astype(np.float32),
            pad=pad,
            multi_ts=ts[multi_local],
            multi_pred=multi_pred.astype(np.float32),
        )

    mse_by_depth = (sq_sum / np.maximum(n_valid, 1)).tolist()
    overall = float(sq_sum.sum() / n_valid.sum())
    metrics = {
        "policy": args.policy,
        "policy_repo": preset["policy_repo"],
        "dataset_repo": preset["dataset_repo"],
        "episodes": episodes,
        "stride": args.stride,
        "k_samples": k_samples,
        "multi_stride": args.multi_stride,
        "horizon": horizon,
        "action_dim": action_dim,
        "overall_mse": overall,
        "overall_rmse": float(np.sqrt(overall)),
        "mse_by_depth": mse_by_depth,
        "rmse_depth_first": float(np.sqrt(mse_by_depth[0])),
        "rmse_depth_last": float(np.sqrt(mse_by_depth[-1])),
        "per_episode": per_episode,
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"done in {metrics['runtime_s']}s: overall RMSE {metrics['overall_rmse']:.3f}, "
        f"RMSE depth 1 -> {horizon}: {metrics['rmse_depth_first']:.3f} -> {metrics['rmse_depth_last']:.3f}"
    )
    print(f"wrote {out_dir}/metrics.json + {len(per_episode)} episode npz files")


if __name__ == "__main__":
    main()
