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
    # Config decisions: NOTES_TRAINING.md (batch 64, 200k steps, seed 100000 match the
    # official lerobot/diffusion_pusht card which reaches ~65% success).
    lerobot-train \
      --policy.type=diffusion \
      --dataset.repo_id=lerobot/pusht \
      --dataset.episodes="[$(seq -s, 0 184)]" \
      --steps=200000 --batch_size=64 --seed=100000 \
      --policy.device=cuda \
      --env.type=pusht --env_eval_freq=25000 --save_freq=25000 \
      --output_dir=outputs/train/diffusion_pusht
    ;;
  act)
    # Episodes 45-49 held out for mock deployment (DESIGN.md §2).
    # Config decisions: NOTES_TRAINING.md (batch 8, 100k steps; ACT converges faster).
    lerobot-train \
      --policy.type=act \
      --dataset.repo_id=lerobot/aloha_sim_insertion_human \
      --dataset.episodes="[$(seq -s, 0 44)]" \
      --steps=100000 --batch_size=8 \
      --policy.device=cuda \
      --env.type=aloha --env_eval_freq=20000 --save_freq=20000 \
      --output_dir=outputs/train/act_aloha
    ;;
  hfjobs-smoke)
    # Cloud smoke test on HF Jobs: short + timeboxed to measure cost/speed before the
    # full run. Requires `hf auth login` first. Pushes a throwaway checkpoint to the Hub.
    lerobot-train \
      --policy.type=diffusion \
      --dataset.repo_id=lerobot/pusht \
      --steps=5000 --batch_size=64 --seed=100000 \
      --policy.repo_id=nilakarthikesan/smoke_diffusion_pusht \
      --policy.push_to_hub=true \
      --job.target=a10g-small --job.timeout=1h
    ;;
  hfjobs-diffusion)
    # Full diffusion run on HF Jobs (see NOTES_TRAINING.md for config rationale).
    lerobot-train \
      --policy.type=diffusion \
      --dataset.repo_id=lerobot/pusht \
      --dataset.episodes="[$(seq -s, 0 184)]" \
      --steps=200000 --batch_size=64 --seed=100000 \
      --env.type=pusht --env_eval_freq=25000 --save_freq=25000 \
      --policy.repo_id=nilakarthikesan/diffusion_pusht \
      --policy.push_to_hub=true --save_checkpoint_to_hub=true \
      --job.target=a100-large --job.timeout=12h
    ;;
  hfjobs-act)
    # Full ACT run on HF Jobs (see NOTES_TRAINING.md for config rationale).
    lerobot-train \
      --policy.type=act \
      --dataset.repo_id=lerobot/aloha_sim_insertion_human \
      --dataset.episodes="[$(seq -s, 0 44)]" \
      --steps=100000 --batch_size=8 \
      --env.type=aloha --env_eval_freq=20000 --save_freq=20000 \
      --policy.repo_id=nilakarthikesan/act_aloha_insertion \
      --policy.push_to_hub=true --save_checkpoint_to_hub=true \
      --job.target=a100-large --job.timeout=12h
    ;;
  *)
    echo "usage: $0 [smoke|diffusion|act|hfjobs-smoke|hfjobs-diffusion|hfjobs-act]" >&2; exit 1
    ;;
esac
