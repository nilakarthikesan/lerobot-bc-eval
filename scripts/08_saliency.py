"""M7 probe: occlusion-sensitivity saliency for a policy's CNN encoder.

Design: NOTES_SALIENCY.md. Slide a grey patch over each held-out frame; at every grid
position re-query the policy and record how far the predicted action chunk moves from the
unoccluded baseline. Large movement = the policy relied on what was under the patch.

This is the ONLY M7 script that runs inference (pipeline rule): it writes arrays that
09_saliency_figures.py consumes without re-inferring.

Usage:
  python scripts/08_saliency.py --policy act_aloha   --steps 020000 100000
  python scripts/08_saliency.py --policy diffusion_pusht --steps 025000 200000
  python scripts/08_saliency.py --policy act_aloha --steps 020000 --episodes 45 \
      --n-frames 6 --patch 96 --stride 48        # quick single-checkpoint probe
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# All checkpoints/datasets are already cached; online snapshot_download hangs on this
# machine's flaky HF connectivity (issue M7-1). Force the cache, no network round-trips.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# LeRobot's revision-safe dataset cache. Passing this as `root` loads purely from disk;
# with root=None the loader probes the Hub for refs, which hangs offline (issue M7-1).
LEROBOT_HUB = Path.home() / ".cache/huggingface/lerobot/hub"

PRESETS = {
    "diffusion_pusht": {
        "policy_cls": DiffusionPolicy,
        "policy_repo": "nilakarthikesan/diffusion_pusht",
        "dataset_repo": "lerobot/pusht",
        "episodes": [190],
        "patch": 16,
        "stride": 8,
        "num_inference_steps": 10,  # match the screen config (03b); full 100 is too slow
    },
    "act_aloha": {
        "policy_cls": ACTPolicy,
        "policy_repo": "nilakarthikesan/act_aloha_insertion",
        "dataset_repo": "lerobot/aloha_sim_insertion_human",
        "episodes": [47],
        "patch": 96,
        "stride": 48,
        "num_inference_steps": None,
    },
}


def checkpoint_path(repo: str, step: str) -> str:
    """Local dir of a banked checkpoint (uses the HF cache; no re-download if present)."""
    snap = snapshot_download(repo, allow_patterns=f"checkpoints/{step}/pretrained_model/*")
    return f"{snap}/checkpoints/{step}/pretrained_model"


def dataset_root(dataset_repo: str) -> str:
    """Concrete on-disk snapshot dir for a cached dataset (offline-safe, no Hub probe)."""
    cache_name = "datasets--" + dataset_repo.replace("/", "--")
    snaps = sorted(glob.glob(str(LEROBOT_HUB / cache_name / "snapshots" / "*")))
    if not snaps:
        raise FileNotFoundError(
            f"{dataset_repo} not in the local LeRobot cache ({LEROBOT_HUB}). Run it once online first.")
    return snaps[-1]


def build_dataset(preset: dict, cfg, episodes: list[int]) -> LeRobotDataset:
    """Dataset windows per the checkpoint's I/O contract (mirrors 03_mock_deploy.py)."""
    root = dataset_root(preset["dataset_repo"])
    fps = LeRobotDataset(preset["dataset_repo"], root=root, episodes=[episodes[0]]).meta.fps
    delta_timestamps = {}
    if cfg.observation_delta_indices is not None:
        deltas = [i / fps for i in cfg.observation_delta_indices]
        delta_timestamps = {key: deltas for key in cfg.input_features}
    delta_timestamps["action"] = [i / fps for i in range(cfg.n_action_steps)]
    return LeRobotDataset(preset["dataset_repo"], root=root, episodes=episodes,
                          delta_timestamps=delta_timestamps)


def to_rgb_uint8(img: torch.Tensor) -> np.ndarray:
    """Last obs step -> HWC uint8 for figure backdrops. Handles [C,H,W] and [T,C,H,W]."""
    frame = img[-1] if img.ndim == 4 else img
    return (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def grid_starts(size: int, patch: int, stride: int) -> list[int]:
    """Patch top/left coords; last one clamped so the patch stays inside the image."""
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def occlude(img: torch.Tensor, y0: int, x0: int, patch: int, grey) -> torch.Tensor:
    """Grey out a patch. Handles [C,H,W] (ACT) and [T,C,H,W] (Diffusion window)."""
    out = img.clone()
    if img.ndim == 3:
        out[:, y0:y0 + patch, x0:x0 + patch] = grey
    else:  # [T, C, H, W] -> same spatial patch in every obs step
        out[:, :, y0:y0 + patch, x0:x0 + patch] = grey
    return out


def predict_chunks(policy, pre, post, image_key, state_key, imgs, state_single, device,
                   noise=None):
    """imgs [N, ...] occluded variants; state repeated to N. Returns [N, H, A] numpy."""
    n = imgs.shape[0]
    rep = state_single.unsqueeze(0).expand(n, *state_single.shape)
    batch = {image_key: imgs, state_key: rep}
    proc = pre(batch)
    proc = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in proc.items()}
    with torch.no_grad():
        if noise is not None:
            chunk = policy.predict_action_chunk(proc, noise=noise[:n])
        else:
            chunk = policy.predict_action_chunk(proc)
    return post(chunk).cpu().numpy()


def saliency_for_frame(policy, pre, post, image_key, state_key, item, device,
                       patch, stride, occ_batch, noise0):
    img = item["image"]
    state = item["state"]
    grey = float(img.mean())

    ys = grid_starts(img.shape[-2], patch, stride)
    xs = grid_starts(img.shape[-1], patch, stride)
    positions = [(y, x) for y in ys for x in xs]

    baseline = predict_chunks(policy, pre, post, image_key, state_key, img.unsqueeze(0),
                              state, device, noise=noise0)  # [1, H, A]

    sal = np.empty(len(positions), dtype=np.float32)
    for b in range(0, len(positions), occ_batch):
        chunk_pos = positions[b:b + occ_batch]
        variants = torch.stack([occlude(img, y, x, patch, grey) for y, x in chunk_pos])
        preds = predict_chunks(policy, pre, post, image_key, state_key, variants, state,
                               device, noise=noise0)  # [n, H, A]
        sal[b:b + len(chunk_pos)] = np.abs(preds - baseline).mean(axis=(1, 2))
    return sal.reshape(len(ys), len(xs)), (ys, xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, choices=sorted(PRESETS))
    ap.add_argument("--steps", nargs="+", required=True, help="checkpoint steps, e.g. 020000 100000")
    ap.add_argument("--episodes", default=None, help="comma-separated override")
    ap.add_argument("--n-frames", type=int, default=8, help="strided frames sampled per episode")
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--occ-batch", type=int, default=16, help="occlusion positions per forward pass")
    ap.add_argument("--seed", type=int, default=42, help="fixes diffusion noise for a stable baseline")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    preset = PRESETS[args.policy]
    episodes = [int(e) for e in args.episodes.split(",")] if args.episodes else preset["episodes"]
    patch = args.patch or preset["patch"]
    stride = args.stride or preset["stride"]
    out_dir = Path(args.output_dir or f"outputs/saliency/{args.policy}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Decode ONCE (per policy): the frame is identical across checkpoints, and both
    # dataset construction (~2min) and per-frame video decode (~20s) are expensive on this
    # machine, so paying them per checkpoint would dominate runtime (issue M7-2). ---
    cfg0 = preset["policy_cls"].from_pretrained(checkpoint_path(preset["policy_repo"], args.steps[0])).config
    image_key = next(k for k in cfg0.input_features if "image" in k)
    state_key = next(k for k in cfg0.input_features if k.endswith("state"))
    ds = build_dataset(preset, cfg0, episodes)
    action_dim = ds[0]["action"].shape[-1]

    epmeta = ds.meta.episodes
    lengths = {int(e): int(t) - int(f) for e, f, t in zip(
        epmeta["episode_index"], epmeta["dataset_from_index"], epmeta["dataset_to_index"])}
    frames = {}  # ep -> list of {image, state, rgb, ts}
    t_dec = time.time()
    for ep in episodes:
        offset = sum(lengths[e] for e in sorted(episodes) if e < ep)
        local = np.linspace(0, lengths[ep] - 1, args.n_frames).astype(int)
        items = []
        for li in local:
            it = ds[offset + int(li)]
            items.append({"image": it[image_key], "state": it[state_key],
                          "rgb": to_rgb_uint8(it[image_key]), "ts": int(li)})
        frames[ep] = items
    print(f"decoded {sum(len(v) for v in frames.values())} frames in {time.time()-t_dec:.0f}s")

    for step in args.steps:
        t0 = time.time()
        path = checkpoint_path(preset["policy_repo"], step)
        policy = preset["policy_cls"].from_pretrained(path)
        policy.eval()
        device = next(policy.parameters()).device
        cfg = policy.config
        if preset["num_inference_steps"] is not None and hasattr(policy, "diffusion"):
            policy.diffusion.num_inference_steps = preset["num_inference_steps"]
        pre, post = make_pre_post_processors(
            cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        # Fixed noise so diffusion's stochasticity does not masquerade as saliency (S1).
        # ACT is deterministic and takes no noise arg, so keep it None there.
        noise0 = None
        if hasattr(policy, "diffusion"):
            gen = torch.Generator().manual_seed(args.seed)
            noise0 = torch.randn(args.occ_batch + 1, cfg.n_action_steps, action_dim,
                                 generator=gen).to(device)

        for ep in episodes:
            maps, ts, rgbs = [], [], []
            for item in frames[ep]:
                sal, _ = saliency_for_frame(
                    policy, pre, post, image_key, state_key, item, device,
                    patch, stride, args.occ_batch, noise0)
                maps.append(sal)
                ts.append(item["ts"])
                rgbs.append(item["rgb"])
            maps = np.stack(maps)  # [F, gh, gw]
            h, w = frames[ep][0]["image"].shape[-2:]

            meta = json.dumps({
                "policy": args.policy, "step": step, "episode": ep, "image_key": image_key,
                "image_hw": [int(h), int(w)], "n_positions": int(maps[0].size),
                "patch": patch, "stride": stride,
                "num_inference_steps": preset["num_inference_steps"],
            })
            np.savez_compressed(
                out_dir / f"step{step}_ep{ep:03d}.npz",
                saliency=maps.astype(np.float32), frames_ts=np.array(ts),
                frames_rgb=np.stack(rgbs), grid=np.array([patch, stride, int(h), int(w)]),
                meta=meta,
            )
            print(f"{args.policy} step {step} ep{ep}: {maps.shape} grid={maps.shape[1]}x{maps.shape[2]} "
                  f"peak/mean={maps.max()/maps.mean():.1f} ({time.time()-t0:.0f}s)")

        del policy
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
