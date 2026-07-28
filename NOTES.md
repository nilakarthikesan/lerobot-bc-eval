# Learning Notes

Running notes kept per stage of the project. Each section records what was understood
at that stage, in our own words, so the final writeup can trace the reasoning.

Stage Three (the pipeline deep-dive: how each remaining stage is implemented and how
they flow together) lives in its own document: [NOTES_PIPELINE.md](NOTES_PIPELINE.md).

---

## Stage One Notes — what a policy is, and how training works

### What a policy is

A **policy** is the function that decides what the robot does: it maps what the robot
currently perceives to the command it should send to its motors. Written as math:
π(a | o) — a (possibly probabilistic) mapping from observations *o* to actions *a*.

- **Observation** = camera image + *proprioception* (the robot's sense of its own body).
  - PushT: a 96×96 image + the 2D position of the pusher.
  - ALOHA: a 480×640 image + the 14 joint angles of the two arms.
- **Action** = the motor command.
  - PushT: a 2D target position (the action *is* the end-effector target).
  - ALOHA: 14 target joint angles, sent 50×/second.

Classical robotics wrote this function by hand (controllers, planners). Learning-based
robotics makes the policy a **neural network**, and "training a policy" = fitting its
weights.

### The two families of policy training

1. **Reinforcement learning** — the robot tries things and learns from a reward signal.
   Powerful, but sample-hungry and risky on real hardware.
2. **Imitation learning** — a human teleoperates the robot through the task a few hundred
   times; every timestep of those recordings becomes a labeled example:
   *"when you saw this, the expert did that."*

We are doing imitation learning, specifically **behavior cloning (BC)**: reduce robot
learning to ordinary supervised learning on (observation, action) pairs.

### What training actually is

The standard deep-learning loop: sample a batch of (observation, action) pairs from the
demonstrations → network predicts actions from observations → loss against the expert's
actions → backpropagate → repeat ~100k times.

Key realization: during training the model is **not practicing the task**. No simulator,
no robot, no reward. It only ever looks at recorded frames and predicts what the human
did. Whether the resulting policy actually *works* is a separate question — that's what
the mock deployment (M4) exists to answer.

### Tools (and how researchers typically do this)

Typical research workflow: teleoperate → record demos in a standard format → train with a
standard recipe → evaluate by success rate over many rollouts.

- **PyTorch** — both models are plain PyTorch modules.
- **LeRobot** — standardizes the pipeline: `LeRobotDataset` (MP4-backed video + Arrow
  tables, streamable from the Hub), reference ACT/Diffusion implementations with paper
  hyperparameters as defaults, and the `lerobot-train` / `lerobot-eval` CLIs. Before
  LeRobot, every lab forked the paper authors' repos (act, diffusion_policy, robomimic).
- **Weights & Biases** — loss curves, run comparison.
- **GPU** — published ballpark: ACT ~2–8 h, Diffusion ~4–16 h on an RTX-4090-class card.

Research habit we copy: **never judge a policy by training loss** — a low loss only says
the network memorized the demos. Papers report success rate over ~50 sim rollouts; that
is exactly our M4 structure, and why `lerobot-train` has `--env_eval_freq` (periodic
rollout eval *during* training).

---

## Stage Two Notes — the task, the two policies, and choosing datasets

### The task, precisely (it is not open-ended)

Train the two most widely used imitation-learning policies — **ACT** and **Diffusion
Policy** — on LeRobot datasets, understand what happens inside each, then **mock deploy**:
hold out a test set the models never saw, run the trained policies on it, and visualize
predicted end-effector trajectories vs. ground truth. Report metrics + figures + any
LeRobot problems encountered.

What "accomplished" means here: **not** state-of-the-art numbers. The deliverable is a
demonstrated understanding — two trained policies, an honest two-part evaluation
(open-loop replay + closed-loop rollout), visualizations that make the policies'
behavior legible, and a writeup that can explain *why* the results look the way they do.

### Why naive BC fails (the reason there are two policies to train)

Naive BC predicts **one action per step** and regresses it with MSE. Two failure modes:

1. **Compounding errors / covariate shift** — tiny per-step errors accumulate; the robot
   drifts into states that never appeared in the demos, where the policy is untrained and
   fails harder, drifting further.
2. **Mode averaging** — when demos contain two valid strategies (push the block from the
   left OR from the right), MSE regression predicts their *average*, which is often an
   invalid action (push straight into the block).

ACT and Diffusion Policy are the two standard fixes, each attacking a different failure:

| | ACT (Zhao et al., RSS 2023) | Diffusion Policy (Chi et al., RSS 2023) |
|---|---|---|
| Attacks | Compounding errors | Mode averaging |
| Core idea | **Action chunking**: predict the next ~100 actions in one forward pass → far fewer decision points at which to drift; temporal ensembling smooths chunk boundaries | **Denoising**: learn to reverse a noising process over action sequences; different noise seeds converge to *different valid strategies* instead of their average |
| Architecture | Transformer trained as a CVAE (encoder only used at training time) | U-Net / transformer denoiser conditioned on the observation |
| Training loss | L1(predicted chunk, expert chunk) + KL regularization on the latent | MSE(predicted noise, actual noise added) |
| Inference | One forward pass (fast, ~ms) | 10–100 iterative denoising steps (slower) |
| Character | Data-efficient, few hyperparameters, fast; CVAE tends to collapse modes | Explicitly multimodal; more tuning (noise schedule, step count), wants more data |

Training mechanics are identical in shape for both (same supervised loop, same
`lerobot-train` CLI); only the loss and the sampling procedure differ. LeRobot fills in
each paper's hyperparameters as defaults, which is the standard research baseline.

### How we chose the datasets (policy ↔ dataset pairing)

Selection criteria, in order:

1. **Match each policy to the task it was designed for**, so results are interpretable
   against published baselines — if our numbers are wildly off, we know it's our
   pipeline, not the method.
2. **A simulator must exist** for the dataset's task, or closed-loop deployment
   (policy actually driving the environment) is impossible and we lose half of M4.
3. **Training must be feasible** with uncertain GPU access (dataset size, image
   resolution, action dimensionality all drive compute).

The pool considered: LeRobot's Hub datasets (`lerobot/pusht`, the `aloha_sim_*` family,
`aloha_mobile_cabinet`, and various real-robot sets).

Chosen pairing:

- **Diffusion Policy ← `lerobot/pusht`** — its own canonical benchmark. The task is
  deliberately **multimodal** (many valid ways to push the T-block), which is exactly the
  property diffusion exists to handle — so the pairing lets us *show* the method's
  signature strength in our visualizations. Small (206 eps, 96×96 images, 2D actions,
  10 fps) → tractable even without a big GPU. `gym-pusht` exists for closed-loop eval.
- **ACT ← `lerobot/aloha_sim_insertion_human`** — the bimanual ALOHA setup ACT was built
  for: fine manipulation at 50 Hz with 14-dim actions, where chunking (predict ~100
  actions ≈ 2 s of motion at once) is the signature move. `gym-aloha` exists for
  closed-loop eval.
- **Rejected as primary: `lerobot/aloha_mobile_cabinet`** (real-robot data shown in the
  meeting) — more impressive visuals but **no simulator**, so evaluation would be
  offline-only. Kept as a stretch goal.

Why the pairing changes training: PushT's low resolution allows batch 64; ALOHA's
480×640 images at batch 8 already stress <8 GB GPUs. ALOHA's 50 fps means a 100-action
chunk spans 2 s of motion (chunk length is defined in *frames*, so fps determines the
time horizon of a chunk). And the M1 split is fixed **before** training so the held-out
episodes (pusht 185–205, aloha 45–49) can never leak into the weights.
