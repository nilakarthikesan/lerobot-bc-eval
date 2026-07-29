# Design Doc — Behavior Cloning with LeRobot: ACT + Diffusion Policy

**Task (as assigned):** Using [LeRobot](https://github.com/huggingface/lerobot), train the
two most widely used imitation-learning policies — **ACT** and **Diffusion Policy** — on
LeRobot datasets, understand what is happening inside each, then do a **mock deployment**:
hold out a test set, run the trained policy on it, and **visualize the model's end-effector
predictions** (Viser can be used here too).

**Deliverable:** Training runs for both policies, a mock-deployment evaluation with
metrics + prediction-vs-ground-truth visualizations, and a short writeup. Flag any LeRobot
issues encountered (she hasn't used the library herself and asked to hear about problems).

---

## 1. Background: the two policies

**ACT (Action Chunking with Transformers)** — a CVAE + transformer that, given camera
images and proprioception, predicts a *chunk* of the next ~100 actions at once. Chunking
plus temporal ensembling smooths out compounding errors, the classic failure mode of
single-step behavior cloning. Introduced with the ALOHA bimanual rig (Zhao et al., 2023).

**Diffusion Policy** — models the action distribution with a conditional denoising
diffusion process: starting from Gaussian noise, it iteratively denoises an action sequence
conditioned on the observation. Its strength is *multimodality* — when demonstrations
contain several valid ways to do the task, regression-style BC averages them (often into an
invalid action) while diffusion can commit to one mode (Chi et al., 2023).

Both are implemented in LeRobot in plain PyTorch (`--policy.type=act` / `diffusion`) and are
exactly the implementations Irmak pointed at in the README's SoTA Models section.

## 2. Datasets and the train/test split

Pick one dataset per policy, matching each policy to the task it was designed for:

| Policy | Dataset | Why |
|--------|---------|-----|
| Diffusion | `lerobot/pusht` | The canonical Diffusion Policy benchmark. Small (~200 episodes, 2D push task) → trains in hours, feasible even without a big GPU. Has a Gym env (`gym-pusht`) for closed-loop rollout. The task is deliberately multimodal (many valid push directions), so it showcases exactly what diffusion is for. |
| ACT | `lerobot/aloha_sim_insertion_human` | The task ACT was built for (bimanual ALOHA, sim). Has `gym-aloha` for closed-loop rollout, so both halves of the mock deployment are possible. |

**Decision:** sim dataset is primary for ACT. `lerobot/aloha_mobile_cabinet` (the real-robot
dataset shown in the meeting) is a **stretch goal** — more impressive visuals, but it has no
sim env, so evaluation there would be offline-only.

**Split:** `LeRobotDataset` is episode-indexed. Reserve the last ~10% of episode indices as
the held-out test set and train with `--dataset.episodes="[0,...,N_train-1]"`. The held-out
episodes are never seen in training and become the "mock deployment" input.

Exact split (fixed from the M1 EDA, used everywhere downstream):

| Dataset | Episodes | Train | Held-out test |
|---------|----------|-------|---------------|
| `lerobot/pusht` | 206 | 0–184 (185) | 185–205 (21) |
| `lerobot/aloha_sim_insertion_human` | 50 | 0–44 (45) | 45–49 (5) |

## 3. Pipeline

```
 stage 0          stage 1            stage 2             stage 3               stage 4
 env setup  ───▶  dataset EDA  ───▶  training      ───▶  mock deployment ───▶  writeup
 (uv/venv,        (episodes,         (lerobot-train,     (offline replay on    (metrics table,
  lerobot,         fps, action        smoke test on       held-out episodes +   plots, videos,
  gym-pusht,       space, sample      Mac MPS, full       closed-loop rollout   issues found)
  gym-aloha)       videos)            runs on GT GPU)     via lerobot-eval)
```

### Stage 0 — Environment
- Python 3.10+, `pip install "lerobot[training,pusht,aloha]==0.6.0"`. As of v0.6.0 the base
  install no longer includes training deps (`training` extra required) and the minimum
  PyTorch is **2.7**. Pin everything in `requirements.txt` for reproducibility.
- Verify dataset streaming: load a few episodes of `lerobot/pusht` and decode video frames.

### Stage 1 — Dataset exploration (`scripts/01_explore_dataset.py`)
- Per dataset: number of episodes/frames, fps, camera keys, state/action dimensionality,
  episode length distribution; dump a few decoded frames and an action-trajectory plot.
- This doubles as the "learn what's happening" evidence for the writeup.

### Stage 2 — Training
Smoke test locally first (Apple Silicon, tiny run, proves the pipeline end to end):

```bash
lerobot-train --policy.type=diffusion --dataset.repo_id=lerobot/pusht \
  --steps=2000 --batch_size=32 --policy.device=mps --output_dir=outputs/smoke_diffusion
```

Full runs on Georgia Tech compute (PACE / AI Makerspace H100s — access confirmation is an
open action item; fallbacks in §6, primarily HF Jobs cloud training):

```bash
# Diffusion on PushT
lerobot-train --policy.type=diffusion --dataset.repo_id=lerobot/pusht \
  --dataset.episodes="[0..184]" --steps=200000 --batch_size=64 --seed=100000 \
  --policy.device=cuda --output_dir=outputs/train/diffusion_pusht \
  --env_eval_freq=25000 --save_freq=25000 --env.type=pusht

# ACT on ALOHA sim insertion
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.episodes="[0..44]" --steps=100000 --batch_size=8 \
  --policy.device=cuda --output_dir=outputs/train/act_aloha \
  --env_eval_freq=20000 --save_freq=20000 --env.type=aloha
```

Track loss curves (WandB via `--wandb.enable=true`, or the local logs). The full
component-by-component rationale for every training hyperparameter is in
[NOTES_TRAINING.md](NOTES_TRAINING.md) (batch/steps/normalization/optimizer decisions).

### Stage 3 — Mock deployment (the core deliverable)
Two complementary evaluations. They answer **different questions**, and the literature is
clear that open-loop action error is a weak predictor of closed-loop success: small errors
compound during rollout and push the policy into states absent from training data
(covariate shift). Recent work even shows policies with *higher* validation MSE achieving
*higher* closed-loop success (arXiv:2604.02523). We therefore report both and treat any
disagreement between them as a finding to discuss in the writeup.

**(a) Offline open-loop replay on held-out episodes** (`scripts/03_mock_deploy.py`)
- For each test episode, step through its observations, query the policy, and record
  predicted action chunks/sequences vs. the ground-truth demonstrator actions.
  (v0.6.0 added offline batch inference support for ACT and Diffusion, which this uses.)
- Metrics: per-step action MSE, end-effector position error over a prediction horizon.
- End-effector extraction: PushT's action *is* the 2D end-effector target, so predictions
  plot directly onto the workspace image. For ALOHA, actions are joint positions → run
  forward kinematics on the ALOHA arm model to get 3D EE positions (or, minimum viable
  version, compare in joint space and visualize per-joint trajectories).

**(b) Closed-loop rollout in sim** (true "deployment": the policy drives the env)
- `lerobot-eval --policy.path=outputs/train/... --env.type=pusht --eval.n_episodes=50`
- Metrics: success rate, episode reward; save rollout videos.

### Stage 4 — Visualization (`scripts/04_visualize.py`)
- **PushT (2D):** overlay predicted vs. ground-truth EE trajectories on the workspace;
  animate the predicted action horizon fanning out ahead of the current state — this shows
  multimodality for diffusion nicely.
- **ALOHA (3D):** Viser scene plotting ground-truth EE trace vs. predicted chunks as
  colored line segments per timestep (reusing Viser skills from the Task-1 repo).
- Static matplotlib versions of everything for the writeup/email.

## 4. Repository layout

```
lerobot-bc-eval/
├── DESIGN.md                    ← this document
├── README.md                    ← quickstart
├── requirements.txt
├── scripts/
│   ├── 01_explore_dataset.py    ← EDA: episodes, action space, sample frames
│   ├── 02_train.sh              ← smoke-test + full lerobot-train commands
│   ├── 03_mock_deploy.py        ← held-out episode replay, metrics
│   └── 04_visualize.py          ← EE prediction vs ground-truth plots / Viser
├── outputs/                     ← checkpoints, eval results (gitignored)
└── reports/                     ← plots, videos, writeup for Irmak
```

## 5. Milestones

| # | Milestone | Definition of done |
|---|-----------|--------------------|
| M0 | Environment | lerobot installed; pusht dataset loads and frames decode |
| M1 | EDA | dataset stats + sample visualizations in `reports/` |
| M2 | Smoke train | 2k-step diffusion run completes on MPS; loss decreases |
| M3 | Full training | ACT + diffusion trained (GT GPU or fallback); loss curves saved |
| M4 | Mock deploy | held-out replay metrics + closed-loop `lerobot-eval` success rates |
| M5 | Visualization | predicted-vs-GT EE trajectory figures (+ rollout videos) |
| M6 | Writeup | metrics table + figures + "issues encountered with LeRobot" notes |

## 6. Compute plan

- **Local (M2 smoke tests):** Apple Silicon MPS, small batch, few thousand steps.
- **Full runs:** Georgia Tech PACE / AI Makerspace (160× H100) / Phoenix cluster — she
  looked these up during the meeting; confirming access is an **open action item**. On an
  H100, both runs are a few hours each.
- **Fallback (concrete, new in v0.6.0):** HF Jobs cloud training — the same `lerobot-train`
  command with `--job.target=a10g-small` runs remotely and pushes the checkpoint to the Hub
  (pay-as-you-go billing). GPU access is therefore never a hard blocker.
- **Last resort:** reduced-step overnight runs on MPS. ACT at batch 8 is feasible on <8 GB;
  diffusion wants more, so PushT (low-res) keeps it tractable.

## 7. Risks & mitigations

- **Video decoding issues** (LeRobot datasets are MP4-backed; codec problems are the most
  common install issue): install `ffmpeg` (done), test decoding in M0 before anything else.
- **API drift** (LeRobot moves fast; v0.6.0 current): pin the version; prefer the
  `lerobot-train`/`lerobot-eval` CLI over internal APIs.
- **MPS quirks** (float64 ops unsupported): known workaround is CPU fallback env var; smoke
  test catches this early.
- **No GPU access confirmed yet:** M0–M2 and all of the offline eval tooling work locally,
  and HF Jobs (§6) provides a pay-as-you-go path that removes the hard dependency on
  Georgia Tech cluster access.
- **ALOHA FK for EE extraction is extra work:** joint-space comparison is an acceptable
  minimum; FK via the ALOHA URDF is the stretch version.

## 8. What gets sent to Irmak

**Decision — the artifact is three things:**

1. **Written report** (`reports/REPORT.md`, exportable to PDF) containing:
   - Loss curves for both policies + final checkpoints (Hub-pushed if allowed).
   - Mock-deployment metrics table: action MSE on held-out episodes, closed-loop success
     rates — and discussion of where the two evaluations disagree (see Stage 3).
   - Figures/videos: predicted vs. ground-truth end-effector trajectories, rollout videos.
   - Notes on anything broken or confusing in LeRobot (she explicitly asked for this).
2. **The public GitHub repo** as the reproducible artifact (scripts + pinned deps).
3. **Viser interactive 3D scene** of predicted vs. ground-truth trajectories as the live
   demo moment.
