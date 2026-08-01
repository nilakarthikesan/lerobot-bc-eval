# Deployment Notes — Stage 3 / M4 (Mock Deployment)

The in-depth design for the stage we are implementing now. Training (M3) is complete —
both policies are on the Hub with banked checkpoints (see
[training/DIFFUSION_POLICY.md](training/DIFFUSION_POLICY.md) and
[training/ACT_POLICY.md](training/ACT_POLICY.md)). This stage turns those artifacts into
**numbers and evidence**: how well do the policies actually perform on data and in
environments they never saw during training?

Like the other stage notes, this doc records the design *before* the code, the decisions
and their rationale, and — in §7 — a **live log of every issue we hit** while implementing
and testing (these feed the M6 writeup).

---

## 0. What this stage answers

Two different questions, deliberately kept separate:

1. **Open-loop:** "When shown a situation a human demonstrator was actually in, does the
   policy propose the same actions the human took?" → prediction accuracy on **held-out
   episodes** (PushT 185–205, ALOHA 45–49).
2. **Closed-loop:** "If the policy is put in charge, does it accomplish the task?" →
   **success rate** when the policy drives the simulator and must live with the
   consequences of its own actions.

The literature is explicit that (1) is a weak predictor of (2) — small errors compound
during rollout and push the policy off the training distribution (covariate shift,
DAgger arXiv:1011.0686; arXiv:2604.02523 shows the rankings can even invert). Any
disagreement between our two numbers is a *finding for the report*, not a bug.

---

## 1. Inputs — what M3 handed us

| | Diffusion / PushT | ACT / ALOHA insertion |
|---|---|---|
| Hub repo | `nilakarthikesan/diffusion_pusht` | `nilakarthikesan/act_aloha_insertion` |
| Final model | repo root (= step 200K) | repo root (= step 100K) |
| Banked checkpoints | every 25K: 8 total | every 20K: 5 total |
| Held-out episodes | `lerobot/pusht` 185–205 (21 eps) | `lerobot/aloha_sim_insertion_human` 45–49 (5 eps) |
| Sim env for closed-loop | `gym-pusht` (`PushT-v0`) | `gym-aloha` (MuJoCo) |

**Loading contract.** Each checkpoint directory is a complete policy: weights
(`model.safetensors`), config, and the fitted **pre/post-processor pipelines**
(normalizer/unnormalizer safetensors). We load via `from_pretrained` and let the saved
processors handle all normalization — *never* normalize manually; hand-rolled
normalization is the classic silent BC bug (see training notes §Component 2).

---

## 2. Part (a) — Open-loop replay (`scripts/03_mock_deploy.py`)

### Algorithm

```
for each held-out episode:
    for t in strided timesteps of the episode:
        obs_t  = episode observations at t (image + state), batched
        pred   = policy's FULL predicted chunk/horizon from obs_t
                 (diffusion: 64-step horizon; ACT: 100-step chunk)
        gt     = episode's ground-truth actions [t : t + horizon]
        record (episode, t, pred, gt)
save raw arrays + metrics to disk for Stage 4
```

Key point: we record the **whole predicted horizon** at each queried state, not just the
next action. That is what lets us measure how error grows with prediction depth and lets
Stage 4 draw the "fans" of predicted futures.

### Metrics (defined precisely so the report is unambiguous)

- **Per-step action MSE:** mean squared error between predicted and GT action at
  prediction depth *d*, averaged over states and episodes → a curve `MSE(d)`,
  `d = 1..horizon`. Expect it to grow with *d*.
- **EE position error:** PushT actions *are* 2-D end-effector targets → report in
  workspace units directly. ALOHA actions are 14 joint positions → baseline metric is
  joint-space L2; EE-space error via MuJoCo FK is a Stage-4 stretch.
- **Diffusion multimodality probe:** at strided states, draw **K=8 samples** with
  different noise seeds; record all of them. Spread across samples ≈ multimodality
  (Stage 4's money shot). ACT is near-deterministic (its KL collapsed → latent ignored;
  see ACT notes), so one sample per state suffices — that asymmetry is itself a finding.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Which checkpoint to replay | the **final** model of each policy (best-by-rollout comes from part (b)) | open-loop replay is diagnostic; keep one consistent reference |
| Timestep stride | every step for metrics; every ~10 steps for the K-sample probe | full-resolution error curve, affordable multimodality probe |
| Diffusion sampler | DDPM-100 for metrics (training-matched), DDIM-10 allowed for the K-sample probe | fidelity where numbers matter, speed where volume matters |
| Batch/inference | batch states through the model where the API allows | v0.6.0 added batch inference for both policies |
| Device | local MPS, CPU fallback env var ready (`PYTORCH_ENABLE_MPS_FALLBACK=1`) | inference is cheap relative to training |
| Output format | one `.npz` per episode (`pred [T,K,H,A]`, `gt [T,H,A]`, `ts`) + one `metrics.json` per policy, under `outputs/mock_deploy/<policy>/` | Stage 4 consumes files, never re-runs inference |

### The action-queue trap (implementation note)

`select_action()` on both policies is built for *control*: it maintains an internal queue
and returns **one action per call**, only re-querying the network when the queue empties.
For replay we need the full chunk from a *specific* state, so we must call the underlying
chunk-prediction path (e.g. `predict_action_chunk()` / the policy's generate method) and
reset any queue state between queried timesteps — otherwise predictions leak across
states and the metrics silently measure the wrong thing. Verified behavior goes in §7.

---

## 3. Part (b) — Closed-loop rollout (`lerobot-eval`)

The policy drives the simulator; nothing is replayed.

### Two-phase checkpoint screening (cost-aware)

Because in-training eval was disabled on HF Jobs (image lacks the gym envs — see
training notes §Component 4), we do checkpoint selection here:

1. **Screen:** every banked checkpoint × a small eval (10 episodes, fixed seeds;
   DDIM-10 for diffusion to keep it fast) → coarse success-rate curve over training steps.
2. **Confirm:** the best 1–2 checkpoints × the full protocol (**50 episodes**, fixed
   seed set; DDPM-100 for diffusion) → the numbers that go in the report.

### Commands (shape of them)

```bash
# PushT / diffusion
lerobot-eval --policy.path=nilakarthikesan/diffusion_pusht \
  --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 \
  --policy.device=mps --output_dir=outputs/eval/diffusion_pusht

# ALOHA / ACT
lerobot-eval --policy.path=nilakarthikesan/act_aloha_insertion \
  --env.type=aloha --env.task=AlohaInsertion-v0 --eval.n_episodes=50 \
  --policy.device=mps --output_dir=outputs/eval/act_aloha
```

(Exact flags to be verified against the installed CLI — logged in §7.)

- **Metrics:** success rate (primary), avg/max episode reward, episode length.
- **Artifacts:** rollout videos (a success and a failure per policy, for the report).
- **Seeds fixed** so screen/confirm runs are comparable.
- Published reference point: official `diffusion_pusht` card reports **~65% success**;
  that is our sanity target, not a hard requirement.

### Expected failure modes to watch

- **Env/dataset mismatch** (obs resolution, fps, action scaling differ between the
  dataset and the gym env) → garbage success rates with no crash. Check obs specs first.
- **ALOHA/MuJoCo on macOS:** headless rendering (GL context) is the usual pain point;
  joint-space eval works even if rendering doesn't.
- **MPS gaps:** any op unsupported on MPS falls back to CPU (env var) or we run CPU.
- **Open-loop good, closed-loop bad:** not a bug — that is covariate shift, report it.

---

## 4. Test plan + results (`scripts/sanity_load_infer.py`)

- [x] **T1 PASS** — both final checkpoints load via `from_pretrained` on the Hub repo id.
  Param counts match the training logs exactly: diffusion **262,709,026**, ACT
  **51,613,582**. Both policies expose `predict_action_chunk()`. (Took two real fixes to
  get here — D2, D3 below.)
- [x] **T2 PASS** — held-out episode 190 (204 frames) loads with the policy's
  `delta_timestamps`; obs windows shaped `(1, 2, 3, 96, 96)` image / `(1, 2, 2)` state;
  full preprocessor → network → postprocessor round trip runs on MPS. (Needed D5.)
- [x] **T3 PASS** — unnormalized predictions land in real workspace coordinates
  (`[169.7, 476.2]` inside the dataset's `[12, 511]`), so the normalization round trip is
  sane, not stuck in `[-1, 1]`.
- [x] **T4 PASS** — `predict_action_chunk()` returns a clean `(1, 32, 2)` chunk. Note:
  it returns the **executed** `n_action_steps=32`, *not* the full 64-step horizon — so
  the `MSE(d)` depth curve tops out at d=32 via this API (fine; that is what actually
  gets executed).
- [x] **T5 PASS** — `lerobot-eval` ran 2 PushT episodes end-to-end on MPS
  (~82 s/episode with DDPM-100), videos saved. **Max rewards 0.987 / 0.990** — the
  policy pushes the T to ~99% target coverage (visually confirmed:
  `reports/m4_smoke/pusht_frames.png`) though the binary success flag didn't trip at
  n=2. Bonus: the eval CLI applies its own device override to the saved processor
  pipeline, so issue D5 only affects *direct API* use, not `lerobot-eval`.
- [x] **T6 PASS (mechanics) / flagged (quality)** — ALOHA + MuJoCo runs fine on macOS
  (no rendering blocker; ~17 s/episode). But **reward was exactly 0.0 in both
  episodes**: the arms move purposefully (coordinated bimanual reach toward peg/socket —
  `reports/m4_smoke/act_frames.png`) yet never complete a grasp. See D7.

Cost data for planning part (b): PushT ≈ 82 s/ep (DDPM-100) → 50-episode confirm ≈
~68 min/checkpoint (DDIM-10 screen ~10× cheaper); ALOHA ≈ 17 s/ep → ~15 min per
50-episode eval. The two-phase screen/confirm plan is affordable locally.

Single-state anecdote from T2: one DDPM sample at one mid-episode state has RMSE ≈ 108 px
vs the demonstrator over 32 steps — meaningless alone (multimodality means a *valid* plan
can differ from the human's), which is exactly why the full replay reports `MSE(d)`
averaged over all states/episodes plus multi-sample spread.

## 5. Issues log (live — feeds M6 writeup)

- **D1 (from M3, context):** `huggingface/lerobot-gpu` image lacks `gym_pusht`/`gym_aloha`
  → in-training eval crashed the first diffusion run; eval moved here to M4. Details in
  [training/NOTES_TRAINING.md](training/NOTES_TRAINING.md).
- **D2 (T1, version skew — the big one):** `from_pretrained` on the diffusion checkpoint
  crashed with `draccus DecodingError: The fields 'gradient_checkpointing' are not valid
  for DiffusionConfig`. Root cause: **the `lerobot-gpu:latest` image is built from
  unreleased git main** (PyPI's latest release *is* our 0.6.0), and main added a
  `gradient_checkpointing` config field — so cloud-trained checkpoints are not loadable
  by any released lerobot. draccus parses strictly (no ignore-unknown-fields). **Fix:**
  stripped the inert field (`False`) from all 9 `config.json` files in the diffusion repo
  (root + 8 checkpoints) via `upload_file`; ACT's config had no foreign fields. Train/
  deploy version skew is a textbook MLOps failure and this is a concrete instance.
- **D3 (T1, missing extra):** local venv had no `diffusers` — `DiffusionPolicy` requires
  the `lerobot[diffusion]` extra, which `requirements.txt` never listed (training ran in
  the cloud, so the gap stayed invisible until local inference). Fixed + requirements
  updated to `lerobot[training,diffusion,pusht,aloha]==0.6.0`. (`gym_pusht`/`gym_aloha`
  *were* already installed locally, so T5/T6 are unblocked.)
- **D4 (tooling, minor):** the venv was created with `uv` and has **no pip** — installs
  must go through `uv pip install --python .venv/bin/python`. Also `make_pre_post_processors`
  lives in `lerobot.policies.factory`, *not* `lerobot.processor`.
- **D5 (T2, device portability):** the saved **preprocessor pipeline hard-codes the
  training device** (`device_processor: {'device': 'cuda'}`) and fails to instantiate on
  a cuda-less machine — unlike the policy config, which auto-falls back cuda→mps with a
  warning. **Fix:** pass
  `preprocessor_overrides={"device_processor": {"device": "mps"}}` to
  `make_pre_post_processors`. Asymmetric fallback behavior = an upstream paper cut worth
  reporting.
- **D6 (design guard, from the docs not a bug):** the pre/post-processors are separate
  pipeline artifacts (`policy_preprocessor.json` + safetensors), not baked into the
  policy module — feeding **raw** observations to `predict_action_chunk()` would produce
  garbage *silently*. The sanity script round-trips through the saved pipelines and
  checks output ranges precisely to catch this class of bug.
- **D7 (T6, open question for the screening):** ACT scored **0.0 reward in both** ALOHA
  smoke episodes despite purposeful, coordinated arm motion (video evidence in
  `reports/m4_smoke/`) — it approaches the peg/socket but appears to miss the grasp.
  Candidate explanations, to be resolved by part (b): (a) plain closed-loop compounding
  error — approach is in-distribution, contact is not; (b) the final 100K checkpoint
  overfits — robomimic predicts an earlier checkpoint (20K–60K) may roll out better;
  (c) something env-side (verify `AlohaInsertion-v0`'s staged reward definition and the
  camera/obs mapping). Action: screen all 5 checkpoints × 10 episodes, watch videos of
  the best; only then debug deeper.

## 6. Interfaces to Stage 4 (visualization)

Stage 4 consumes only files produced here — it never re-runs inference:

```
outputs/mock_deploy/<policy>/
├── ep<id>.npz         # ts, pred [T,K,H,A], gt [T,H,A]
└── metrics.json       # MSE(d) curve, EE-error summary, per-episode stats
outputs/eval/<policy>/ # lerobot-eval outputs: success rates, videos
```

## 7. Papers

- DAgger — Ross et al., 2011, arXiv:1011.0686 (compounding errors / covariate shift).
- "Tune to Learn" — arXiv:2604.02523 (open-loop MSE can rank policies opposite to
  closed-loop success; the reason we report both).
- Implicit BC — Florence et al., 2021, arXiv:2109.00137 (PushT origin + multimodality).
- robomimic — arXiv:2108.03298 (checkpoint selection by rollout, the rule part (b)
  implements).
