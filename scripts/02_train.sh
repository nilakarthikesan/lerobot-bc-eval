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
    # Full run (GPU). Episodes 185-205 held out for mock deployment (DESIGN.md §2).
    lerobot-train \
      --policy.type=diffusion \
      --dataset.repo_id=lerobot/pusht \
      --dataset.episodes="[$(seq -s, 0 184)]" \
      --steps=100000 --batch_size=64 \
      --policy.device=cuda \
      --env.type=pusht --env_eval_freq=10000 \
      --output_dir=outputs/train/diffusion_pusht
    ;;
  act)
    # Episodes 45-49 held out for mock deployment (DESIGN.md §2).
    lerobot-train \
      --policy.type=act \
      --dataset.repo_id=lerobot/aloha_sim_insertion_human \
      --dataset.episodes="[$(seq -s, 0 44)]" \
      --steps=100000 --batch_size=8 \
      --policy.device=cuda \
      --env.type=aloha --env_eval_freq=10000 \
      --output_dir=outputs/train/act_aloha
    ;;
  *)
    echo "usage: $0 [smoke|diffusion|act]" >&2; exit 1
    ;;
esac
