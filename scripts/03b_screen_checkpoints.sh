#!/usr/bin/env bash
# Closed-loop checkpoint screen — NOTES_DEPLOYMENT.md §3 part (b), phase 1.
#
# lerobot-eval with a Hub repo id only loads the ROOT (final) model, so each banked
# checkpoint (checkpoints/<step>/pretrained_model/ in the repo) is downloaded first and
# evaluated from its local path. Fixed seed so the curve across checkpoints is comparable.
#
# usage:
#   ./scripts/03b_screen_checkpoints.sh act        [n_episodes=10]  # 5 ckpts, ALOHA
#   ./scripts/03b_screen_checkpoints.sh diffusion  [n_episodes=10]  # 8 ckpts, PushT
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1

POLICY=${1:?usage: $0 [act|diffusion] [n_episodes]}
N_EPISODES=${2:-10}

case "$POLICY" in
  act)
    REPO=nilakarthikesan/act_aloha_insertion
    STEPS="020000 040000 060000 080000 100000"
    ENV_ARGS=(--env.type=aloha --env.task=AlohaInsertion-v0)
    EXTRA=()
    ;;
  diffusion)
    REPO=nilakarthikesan/diffusion_pusht
    STEPS="025000 050000 075000 100000 125000 150000 175000 200000"
    ENV_ARGS=(--env.type=pusht)
    # 10 denoising steps (vs training's 100) makes the screen ~10x cheaper; the confirm
    # run on the winning checkpoint goes back to the full 100 (NOTES_DEPLOYMENT.md §3).
    EXTRA=(--policy.num_inference_steps=10)
    ;;
  *) echo "unknown policy: $POLICY" >&2; exit 1 ;;
esac

for STEP in $STEPS; do
  OUT="outputs/eval/screen_${POLICY}/${STEP}"
  if [[ -f "$OUT/eval_info.json" ]]; then
    echo "== $STEP already evaluated, skipping"
    continue
  fi
  echo "== checkpoint $STEP"
  SNAP=$(.venv/bin/hf download "$REPO" --include "checkpoints/${STEP}/pretrained_model/*")
  SNAP=${SNAP#path=}  # current hf CLI prints "path=<dir>", not the bare dir (issue D8)
  .venv/bin/lerobot-eval \
    --policy.path="${SNAP}/checkpoints/${STEP}/pretrained_model" \
    --policy.device=mps \
    "${ENV_ARGS[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --eval.n_episodes="$N_EPISODES" --eval.batch_size=1 \
    --seed=42 \
    --output_dir="$OUT"
done

echo "== screen complete: success-rate curve =="
for STEP in $STEPS; do
  F="outputs/eval/screen_${POLICY}/${STEP}/eval_info.json"
  [[ -f "$F" ]] || continue
  .venv/bin/python - "$STEP" "$F" <<'PY'
import json, sys
step, path = sys.argv[1], sys.argv[2]
o = json.load(open(path))["overall"]
print(f"step {step}: success {o['pc_success']:.0f}%  avg_max_reward {o['avg_max_reward']:.3f}  avg_sum_reward {o['avg_sum_reward']:.1f}")
PY
done
