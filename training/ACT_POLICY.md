# ACT — ALOHA Sim Insertion (Deep-Dive)

Standalone record for our **ACT** (Action Chunking with Transformers) policy trained on
`lerobot/aloha_sim_insertion_human`. A sub-document of
[NOTES_TRAINING.md](NOTES_TRAINING.md); read that first for the shared machinery
(dataloader, normalization, training loop) and cross-policy rationale. This file focuses on
what is specific to *this* policy.

---

## Status

| | |
|---|---|
| HF Jobs job | `6a69a3b6a9f4e0ab00b2c84d` ([job page](https://huggingface.co/jobs/nilakarthikesan/6a69a3b6a9f4e0ab00b2c84d)) |
| Model repo | [nilakarthikesan/act_aloha_insertion](https://huggingface.co/nilakarthikesan/act_aloha_insertion) |
| Hardware | `a100-large` (HF Jobs) |
| State | **Training in progress** — final results tables below are filled on completion |
| Expected wall time | lighter than diffusion (batch 8, 100K steps) → a few hours |

---

## 1. What this policy is

ACT (Zhao et al., RSS 2023, [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)) predicts a
**chunk** of future actions in one shot with a **transformer**, and wraps it in a
**conditional VAE (CVAE)**. Two ideas do the work:

- **Action chunking.** Rather than one action per step, the policy outputs the next
  `chunk_size=100` actions at once. Committing to a chunk reduces the number of decision
  points, which directly attacks **compounding error** (covariate shift) — the failure mode
  where small per-step mistakes snowball a closed-loop rollout into unseen states.
- **CVAE latent.** A latent variable `z` (encoded from the demo action sequence at train
  time, sampled from the prior at test time) lets the policy represent **multiple valid
  styles** of executing a motion instead of averaging them.

**Why it fits ALOHA insertion.** Bimanual peg-in-hole insertion is high-frequency (50 fps),
contact-rich, and precision-sensitive; per-step regression accumulates error badly. Chunking
+ the CVAE is exactly the recipe ACT was designed for, and it's the reference policy for
ALOHA sim tasks in LeRobot.

---

## 2. Task & data

- **Dataset:** `lerobot/aloha_sim_insertion_human` — 50 episodes of human-teleoperated
  bimanual insertion, multiple camera views, 50 fps.
- **Observation:** RGB image(s) + robot joint state.
- **Action:** target joint positions for both arms.
- **Split:** episodes **0–44 for training** (45 eps); **45–49 held out** for mock deployment
  (M4). `--dataset.episodes="0..44"` keeps the held-out episodes out of training and out of
  the normalization statistics.

---

## 3. Architecture (as configured)

| Block | Setting | Notes |
|-------|---------|-------|
| Vision backbone | ResNet-18, ImageNet-pretrained | |
| Transformer | `dim_model=512`, `n_heads=8`, `dim_feedforward=3200` | |
| Encoder / decoder layers | 4 / **1** | decoder=1 deliberately matches an upstream ACT quirk (only the first decoder layer was used); kept for parity |
| CVAE | `use_vae=True`, `latent_dim=32`, `n_vae_encoder_layers=4` | the multimodality mechanism |
| Chunk | `chunk_size=100`, `n_action_steps=100`, `n_obs_steps=1` | predict + execute a 100-step chunk (2 s at 50 fps) from the current frame |
| Temporal ensembling | off (`temporal_ensemble_coeff=None`) | see decision below |

**Normalization (this policy):** VISUAL = ImageNet mean/std; STATE and ACTION = **MEAN_STD**.
No sample clipping (it's a CVAE, not a denoiser), and joint-space actions are roughly
Gaussian, so standardization is the natural fit.

**Loss.** Reconstruction (L1 on the action chunk) + `kl_weight=10.0` × KL on the latent.
**Watch the KL term** during training — if it collapses toward 0 the latent is being ignored
(the CVAE degenerates to a plain regressor).

---

## 4. Exact run configuration

The command we launched (from `../scripts/02_train.sh`, mode `hfjobs-act`):

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.episodes="$(seq -s, 0 44)" \
  --steps=100000 --batch_size=8 \
  --env.type=aloha --env_eval_freq=20000 --save_freq=20000 \
  --policy.repo_id=nilakarthikesan/act_aloha_insertion \
  --policy.push_to_hub=true --save_checkpoint_to_hub=true \
  --job.target=a100-large --job.timeout=12h
```

| Knob | Value | Why |
|------|-------|-----|
| steps | 100,000 | LeRobot default; ACT is data-efficient and converges faster than diffusion |
| batch_size | 8 | VRAM-bound: 480×640 multi-cam images at batch 8 already need ~12 GB |
| seed | 1000 | default; fixed for reproducibility |
| optimizer | AdamW, lr 1e-5, lr_backbone 1e-5, wd 1e-4 | policy preset; **no LR scheduler** for ACT |
| eval / save freq | every 20,000 steps | banks checkpoints for rollout-based selection |

---

## 5. Training dynamics

Track both loss terms:

| Step | Total loss | L1 recon | KL | Notes |
|------|-----------|----------|----|----|
| _fill from final log_ | | | | KL should stay > 0 |

- Full loss curves + periodic success metric: _attach on completion._

---

## 6. Checkpoints & selection

`save_freq=20000` → checkpoints at 20K, 40K, …, 100K, each pushed to the Hub.

**Selection rule = by closed-loop rollout success, not lowest loss** (same rationale as the
diffusion doc; robomimic [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)). Use the
in-training `env_eval_freq` success curve to pick the candidate, confirm in M4.

| Checkpoint | In-training eval success | Selected? |
|-----------|--------------------------|-----------|
| _20K … 100K_ | _fill from logs_ | _fill_ |

---

## 7. Evaluation (to complete in M3/M4)

- **Closed-loop (primary):** run the selected checkpoint in the ALOHA sim insertion env,
  report success rate over N seeds.
- **Open-loop:** action-prediction L1/MSE vs held-out episodes 45–49.

| Metric | Value |
|--------|-------|
| Closed-loop success | _TBD_ |
| Open-loop action error (held-out) | _TBD_ |

---

## 8. Known quirks / candidate ablations

- **Temporal ensembling.** Default off (with `n_action_steps=100`). Original ACT enables it
  at coeff 0.01, but that forces `n_action_steps=1` (re-query every step → ~100× more
  inference calls). Primary run keeps it **off** to match LeRobot's reference eval and keep
  rollouts fast; **on (coeff 0.01)** is listed as a Stage-4 smoothness ablation.
- **Decoder = 1 layer** parity quirk — could ablate to 4+ layers.

## References

- ACT / ALOHA — Zhao et al., RSS 2023, [arXiv:2304.13705](https://arxiv.org/abs/2304.13705).
- robomimic (checkpoint selection) — Mandlekar et al., [arXiv:2108.03298](https://arxiv.org/abs/2108.03298).
