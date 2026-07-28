# lerobot-bc-eval

Behavior cloning with [LeRobot](https://github.com/huggingface/lerobot): train **ACT** and
**Diffusion Policy** on LeRobot datasets, then run a mock deployment on held-out episodes
and visualize the policies' end-effector predictions against ground truth.

See [DESIGN.md](DESIGN.md) for the full system design and [NOTES.md](NOTES.md) for
per-stage learning notes.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stage 1: explore the datasets
python scripts/01_explore_dataset.py --repo-id lerobot/pusht

# Stage 2: smoke-test training locally (Apple Silicon)
lerobot-train --policy.type=diffusion --dataset.repo_id=lerobot/pusht \
  --steps=2000 --batch_size=32 --policy.device=mps --output_dir=outputs/smoke_diffusion
```

Full training runs target Georgia Tech GPUs (PACE / AI Makerspace) — see DESIGN.md §6.

## Status

- [x] M0 environment; pusht dataset loads + video decodes (lerobot 0.6.0, torch 2.11, Python 3.12)
- [x] M1 dataset EDA in `reports/eda/` (split fixed: pusht 0-184 train / 185-205 test; aloha 0-44 / 45-49)
- [ ] M2 local smoke training run (MPS)
- [ ] M3 full ACT + diffusion training runs
- [ ] M4 mock deployment: held-out replay + closed-loop `lerobot-eval`
- [ ] M5 EE prediction visualizations
- [ ] M6 writeup for Irmak (incl. LeRobot issues encountered)
