#!/usr/bin/env bash
# Training commands (DESIGN.md stage 2). Run the smoke test first.
set -euo pipefail

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    # Local Apple Silicon sanity check: proves the pipeline end to end.
    lerobot-train \
      --policy.type=diffusion \
      --dataset.repo_id=lerobot/pusht \
      --steps=2000 --batch_size=32 \
      --policy.device=mps \
      --output_dir=outputs/smoke_diffusion
    ;;
  diffusion)
    # Full run (GPU). Hold out the last ~10% of episodes for mock deployment.
    lerobot-train \
      --policy.type=diffusion \
      --dataset.repo_id=lerobot/pusht \
      --steps=100000 --batch_size=64 \
      --policy.device=cuda \
      --env.type=pusht --eval_freq=10000 \
      --output_dir=outputs/train/diffusion_pusht
    ;;
  act)
    lerobot-train \
      --policy.type=act \
      --dataset.repo_id=lerobot/aloha_sim_insertion_human \
      --steps=100000 --batch_size=8 \
      --policy.device=cuda \
      --env.type=aloha --eval_freq=10000 \
      --output_dir=outputs/train/act_aloha
    ;;
  *)
    echo "usage: $0 [smoke|diffusion|act]" >&2; exit 1
    ;;
esac
