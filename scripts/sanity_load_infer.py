"""M4 pre-implementation sanity test (NOTES_DEPLOYMENT.md §4, T2-T4).

Loads the trained diffusion policy from the Hub, feeds it one observation window
from a held-out PushT episode through the saved pre/post-processors, and checks
that the predicted action chunk is sane (shape, dtype, range vs dataset stats).

Usage: .venv/bin/python scripts/sanity_load_infer.py
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

REPO = "nilakarthikesan/diffusion_pusht"
HELD_OUT_EPISODE = 190  # train split was 0-184


def main() -> None:
    policy = DiffusionPolicy.from_pretrained(REPO)
    policy.eval()
    device = next(policy.parameters()).device
    print(f"policy on {device}, params {sum(p.numel() for p in policy.parameters()):,}")

    # The saved pipeline hard-codes the training device (cuda); override for local
    # hardware (NOTES_DEPLOYMENT.md issue D5).
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=REPO,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Build the dataset exactly the way training did: delta_timestamps from the
    # policy config, so each item is already a (n_obs_steps obs, horizon actions) window.
    meta_fps = 10  # lerobot/pusht
    delta_timestamps = {
        "observation.image": [i / meta_fps for i in policy.config.observation_delta_indices],
        "observation.state": [i / meta_fps for i in policy.config.observation_delta_indices],
        "action": [i / meta_fps for i in policy.config.action_delta_indices],
    }
    ds = LeRobotDataset("lerobot/pusht", episodes=[HELD_OUT_EPISODE], delta_timestamps=delta_timestamps)
    print(f"held-out episode {HELD_OUT_EPISODE}: {len(ds)} frames")

    item = ds[10]  # a mid-episode state
    batch = {
        "observation.image": item["observation.image"].unsqueeze(0),
        "observation.state": item["observation.state"].unsqueeze(0),
    }
    print("obs image window:", tuple(batch["observation.image"].shape))
    print("obs state window:", tuple(batch["observation.state"].shape))

    # T2: preprocessor -> network -> postprocessor round trip.
    proc_batch = pre(batch)
    proc_batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in proc_batch.items()
    }
    with torch.no_grad():
        chunk = policy.predict_action_chunk(proc_batch)  # T4: full chunk, not queued single actions
    actions = post(chunk)
    print("predicted chunk shape:", tuple(actions.shape))

    # T3: unnormalized predictions must land in the dataset's action range.
    stats = ds.meta.stats["action"]
    amin, amax = torch.as_tensor(stats["min"]), torch.as_tensor(stats["max"])
    a = actions.squeeze(0).cpu()
    print(f"pred action range: [{a.min():.1f}, {a.max():.1f}]  dataset range: [{amin.min():.1f}, {amax.max():.1f}]")
    in_range = bool((a >= amin.min() - 50).all() and (a <= amax.max() + 50).all())

    # Compare against ground truth over the executed part of the horizon.
    gt = item["action"][: a.shape[0]]
    mse = torch.mean((a[: gt.shape[0]] - gt) ** 2).item()
    print(f"MSE vs ground truth over first {gt.shape[0]} steps: {mse:.2f} (workspace units^2)")

    assert actions.ndim == 3 and actions.shape[-1] == 2, "expected [B, steps, 2] action chunk"
    assert in_range, "predictions far outside dataset action range - normalization bug?"
    print("T2-T4 PASS")


if __name__ == "__main__":
    main()
