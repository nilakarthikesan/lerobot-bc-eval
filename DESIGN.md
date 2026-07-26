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
| Diffusion | `lerobot/pusht` | The canonical Diffusion Policy benchmark. Small (~200 episodes, 2D push task) → trains in hours, feasible even without a big GPU. Has a Gym env (`gym-pusht`) for closed-loop rollout. |
| ACT | `lerobot/aloha_sim_insertion_human` | The task ACT was built for (bimanual ALOHA, sim). Has `gym-aloha` for closed-loop rollout. Real-robot alternative: `lerobot/aloha_mobile_cabinet` (the one shown in the meeting) — no sim env, so evaluation is offline-only. |

**Split:** `LeRobotDataset` is episode-indexed. Reserve the last ~10% of episode indices as
the held-out test set and train with `--dataset.episodes="[0,...,N_train-1]"`. The held-out
episodes are never seen in training and become the "mock deployment" input.

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
- Python 3.10+, `pip install "lerobot[pusht,aloha]"` (pulls gym envs). Pin the version in
  `requirements.txt` for reproducibility.
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
open action item; fallback: Colab/Lightning GPU or an overnight MPS run at reduced steps):

```bash
# Diffusion on PushT
lerobot-train --policy.type=diffusion --dataset.repo_id=lerobot/pusht \
  --dataset.episodes="[0..183]" --steps=100000 --batch_size=64 \
  --policy.device=cuda --output_dir=outputs/train/diffusion_pusht \
  --eval_freq=10000 --env.type=pusht

# ACT on ALOHA sim insertion
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.episodes="[0..44]" --steps=100000 --batch_size=8 \
  --policy.device=cuda --output_dir=outputs/train/act_aloha \
  --eval_freq=10000 --env.type=aloha
```

Track loss curves (WandB via `--wandb.enable=true`, or the local logs).

### Stage 3 — Mock deployment (the core deliverable)
Two complementary evaluations:

**(a) Offline open-loop replay on held-out episodes** (`scripts/03_mock_deploy.py`)
- For each test episode, step through its observations, query the policy, and record
  predicted action chunks/sequences vs. the ground-truth demonstrator actions.
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
- **Fallback:** reduced-step overnight runs on MPS, or a cloud GPU notebook. ACT at batch 8
  is feasible on <8 GB; diffusion wants more, so PushT (low-res) keeps it tractable.

## 7. Risks & mitigations

- **Video decoding issues** (LeRobot datasets are MP4-backed; codec problems are the most
  common install issue): install `ffmpeg` (done), test decoding in M0 before anything else.
- **API drift** (LeRobot moves fast; v0.6.0 current): pin the version; prefer the
  `lerobot-train`/`lerobot-eval` CLI over internal APIs.
- **MPS quirks** (float64 ops unsupported): known workaround is CPU fallback env var; smoke
  test catches this early.
- **No GPU access confirmed yet:** M0–M2 and all of the offline eval tooling work locally,
  so training compute is the only blocker and it is parallelizable with everything else.
- **ALOHA FK for EE extraction is extra work:** joint-space comparison is an acceptable
  minimum; FK via the ALOHA URDF is the stretch version.

## 8. What gets sent to Irmak

1. Loss curves for both policies + final checkpoints (Hub-pushed if allowed).
2. Mock-deployment metrics: action MSE on held-out episodes, closed-loop success rates.
3. Figures/videos: predicted vs. ground-truth end-effector trajectories, rollout videos.
4. Notes on anything broken or confusing in LeRobot (she explicitly asked for this).
