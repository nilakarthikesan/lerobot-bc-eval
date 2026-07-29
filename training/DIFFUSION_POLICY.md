# Diffusion Policy — PushT (Deep-Dive)

Standalone record for our **Diffusion Policy** trained on `lerobot/pusht`. A sub-document
of [NOTES_TRAINING.md](NOTES_TRAINING.md); read that first for the shared machinery
(dataloader, normalization, training loop) and cross-policy rationale. This file focuses on
what is specific to *this* policy: what it is, the exact configuration we ran, how training
behaved, which checkpoint we selected, and how it evaluates.

---

## Status

| | |
|---|---|
| HF Jobs job | `6a699a7f15e81eca66a8da3d` ([job page](https://huggingface.co/jobs/nilakarthikesan/6a699a7f15e81eca66a8da3d)) |
| Model repo | [nilakarthikesan/diffusion_pusht](https://huggingface.co/nilakarthikesan/diffusion_pusht) |
| Hardware | `a100-large` (HF Jobs) |
| State | **Training in progress** — final results tables below are filled on completion |
| Measured throughput | ~9.1 step/s (~585 samples/s) at batch 64 → ~6 h for 200K steps |

---

## 1. What this policy is

Diffusion Policy (Chi et al., RSS 2023, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137))
frames action prediction as **conditional denoising diffusion**. Instead of regressing a
single action, it starts from Gaussian noise and iteratively denoises it into a *sequence*
of future actions, conditioned on recent observations. The network is trained to predict
the noise added at a random diffusion timestep (the `epsilon` objective).

**Why it fits PushT.** PushT (push a T-shaped block to a target pose) is *multimodal*: from
many states there are several equally-valid ways to push, and the demonstrations reflect
that. A unimodal regressor (naive BC) averages those modes and produces an action that
pushes toward neither — it stalls. Diffusion represents a full action *distribution*, so it
can commit to one coherent mode per rollout instead of averaging. This is the core reason
we picked it for this task (see [../NOTES.md](../NOTES.md), Stage 2).

---

## 2. Task & data

- **Dataset:** `lerobot/pusht` — 206 episodes, ~25k frames, single top-down camera at 10 fps.
- **Observation:** one RGB image + 2-D agent (end-effector) position.
- **Action:** 2-D target position for the pusher.
- **Split:** episodes **0–184 for training** (185 eps); **185–205 held out** for the mock
  deployment / evaluation stage (M4). We pass `--dataset.episodes="0..184"` so the held-out
  episodes never touch training *or* the normalization statistics.

---

## 3. Architecture (as configured)

| Block | Setting | Notes |
|-------|---------|-------|
| Vision backbone | ResNet-18, ImageNet-pretrained | one encoder (PushT has a single camera) |
| GroupNorm | off (`use_group_norm=False`) | 0.6.0 default; classic DP used GroupNorm — flagged as an ablation |
| Crop | none | 0.6.0 default (classic DP random-cropped 96→84) — flagged as ablation |
| Denoiser | 1-D conditional U-Net, `down_dims=(512,1024,2048)`, `kernel_size=5`, `n_groups=8` | conditioned on obs features via FiLM |
| Horizon | `horizon=64`, `n_action_steps=32`, `n_obs_steps=2` | predict 64 steps (6.4 s), execute 32, condition on 2 frames |
| Noise schedule | DDPM, `num_train_timesteps=100`, `beta_schedule=squaredcos_cap_v2` | iDDPM cosine schedule the DP paper found best |
| Prediction target | `epsilon` (predict the noise) | |
| Sample clipping | `clip_sample=True`, range `[-1,1]` | **pairs with MIN_MAX action normalization** |

**Normalization (this policy):** VISUAL = ImageNet mean/std; STATE and ACTION = **MIN_MAX**
to `[-1, 1]`. This is deliberate and coupled to `clip_sample`: the denoiser clips every
intermediate sample to `[-1, 1]`, which is only valid if actions were scaled into that
range. MEAN_STD here would push values outside the clip range and corrupt sampling.

---

## 4. Exact run configuration

The command we launched (from `../scripts/02_train.sh`, mode `hfjobs-diffusion`):

```bash
lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --dataset.episodes="$(seq -s, 0 184)" \
  --steps=200000 --batch_size=64 --seed=100000 \
  --env.type=pusht --env_eval_freq=25000 --save_freq=25000 \
  --policy.repo_id=nilakarthikesan/diffusion_pusht \
  --policy.push_to_hub=true --save_checkpoint_to_hub=true \
  --job.target=a100-large --job.timeout=12h
```

| Knob | Value | Why |
|------|-------|-----|
| steps | 200,000 | official card's budget; best checkpoint historically ~175K |
| batch_size | 64 | official card value; diffusion wants many noise samples/update |
| seed | 100000 | matches the official `lerobot/diffusion_pusht` card for comparability |
| optimizer | Adam 1e-4, betas (0.95, 0.999), wd 1e-6 | policy preset (published recipe) |
| scheduler | cosine, 500-step warmup | policy preset |
| eval / save freq | every 25,000 steps | banks multiple checkpoints for rollout-based selection |

---

## 5. Training dynamics

Loss is the denoising MSE (predicted vs actual noise), so absolute values are small; the
*trend* is what matters.

| Step | Loss (observed) |
|------|-----------------|
| 2K | ~0.032 |
| 5K | ~0.023 |
| _… fill from final log …_ | |

- **Throughput:** ~0.067 s compute + ~0.043 s data per step on the A100 (dataloader is not
  the bottleneck).
- Full loss curve + the periodic success metric: _attach on completion (from job logs / the
  loss plot for the report)._

---

## 6. Checkpoints & selection

`save_freq=25000` → checkpoints at 25K, 50K, …, 200K, each pushed to the Hub.

**Selection rule = by closed-loop rollout success, not by lowest loss** (per robomimic,
[arXiv:2108.03298](https://arxiv.org/abs/2108.03298); the official card's best was 175K, not
the last step). We read the in-training `env_eval_freq` success curve to pick the candidate,
then confirm it in the M4 mock-deployment stage.

| Checkpoint | In-training eval success | Selected? |
|-----------|--------------------------|-----------|
| _25K … 200K_ | _fill from logs_ | _fill_ |

---

## 7. Evaluation (to complete in M3/M4)

- **Closed-loop (primary):** run the selected checkpoint in the PushT sim, report success
  rate over N seeds. Target ≈ **65%** (published card).
- **Inference sampler:** DDPM-100 primary (sim, not real-time-constrained → favor fidelity);
  **DDIM-10 as an ablation** (~10× faster, ≈ same quality per the DP paper) — this also makes
  the multi-sample multimodality visualization (Stage 4) cheap.
- **Open-loop:** action-prediction MSE vs held-out episodes 185–205 (a sanity check; note the
  open-loop/closed-loop gap discussed in the design doc).

| Metric | Value |
|--------|-------|
| Closed-loop success (DDPM-100) | _TBD_ |
| Closed-loop success (DDIM-10) | _TBD_ |
| Open-loop action MSE (held-out) | _TBD_ |

---

## 8. Known quirks / candidate ablations

- **GroupNorm off** and **no crop** (0.6.0 defaults) vs the classic DP recipe — either could
  be toggled as an ablation.
- **DDIM-10 vs DDPM-100** at inference — speed/quality trade-off.
- All are nice-to-haves for the writeup, not required for the deliverable.

## References

- Diffusion Policy — Chi et al., RSS 2023, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137).
- robomimic (checkpoint selection) — Mandlekar et al., [arXiv:2108.03298](https://arxiv.org/abs/2108.03298).
- Official model card: `lerobot/diffusion_pusht` (batch 64 / 200K / seed 100000 → ~65% @ 175K).
