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
| Timestep stride | ACT every step; **diffusion every 2nd step** (updated after measuring) | DDPM-100 on the 262M UNet costs seconds/state on MPS — stride 1 would take hours. `MSE(d)` is an average, so halving the sample count doesn't bias it |
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
states and the metrics silently measure the wrong thing. **Verified:**
`predict_action_chunk()` has an explicit offline mode (empty queues → uses the batch
directly), so batched replay is safe as long as we never call `select_action()`.

### Implementation design (verified against the trained checkpoints)

**Per-policy I/O contract** (read from the checkpoint configs, not assumed):

| | Diffusion / PushT | ACT / ALOHA |
|---|---|---|
| obs keys | `observation.image` (3,96,96), `observation.state` (2) | `observation.images.top` (3,480,640), `observation.state` (14) |
| obs window (`observation_delta_indices`) | `[-1, 0]` → shape `(B, 2, ...)` | `None` → single frame `(B, ...)` |
| chunk returned | 32 steps | 100 steps |
| **GT alignment** | horizon is generated for deltas −1..62 but the code returns slice `[n_obs_steps-1 : n_obs_steps-1+32]` → **chunk step d = GT action at t+d** | deltas 0..99 → **chunk step d = GT action at t+d** |

So the replay builds its GT window with `action` deltas `[0 .. n_action_steps-1]`
(independent of the policy's internal training horizon) and compares 1:1 with the chunk.

**Data flow** (`scripts/03_mock_deploy.py`):

1. Load policy (`from_pretrained`) + saved pre/post-processors
   (`make_pre_post_processors` with the `device_processor` override, issue D5).
2. Build ONE `LeRobotDataset` over all held-out episodes with `delta_timestamps` derived
   from the policy config: obs keys at `observation_delta_indices` (or `[0]` if `None`),
   `action` at `[0 .. n_action_steps-1] / fps`.
3. Segment episodes via `meta.episodes` `dataset_from_index / dataset_to_index`
   (the 0.6.0 API; `episode_data_index` is gone — same migration as the EDA script).
4. Stack items into batches of query states → preprocess → `predict_action_chunk` →
   postprocess → compare to the GT window.
5. **Padding mask:** items near episode ends have copy-padded GT (`action_is_pad`);
   masked steps are excluded from metrics — otherwise `MSE(d)` at high d is biased by
   frozen GT.
6. Diffusion multimodality probe: at every `--multi-stride`-th state, run the chunk
   prediction K times (fresh noise each call → different samples).
7. Write per-episode `.npz` (`ts`, `pred`, `gt`, `pad_mask`, `multi_pred`) +
   per-policy `metrics.json` (`mse_by_depth`, per-episode summaries, config echo).

**CLI:** `--policy {diffusion_pusht|act_aloha}` (maps to repo + dataset + episodes),
`--episodes` override, `--batch-size`, `--k-samples`, `--multi-stride`, `--device`,
`--output-dir`. Defaults reproduce the report numbers.

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

### Screening runner (`scripts/03b_screen_checkpoints.sh`)

`lerobot-eval --policy.path` accepts a Hub repo id, but that only loads the **root**
(final) model. The banked checkpoints live in `checkpoints/<step>/pretrained_model/`
subdirectories, so the screen must `hf download` each checkpoint's subtree and point
`--policy.path` at the *local* directory. The runner loops
`checkpoint × lerobot-eval(10 episodes, seed=42)` and drops each result in
`outputs/eval/screen_<policy>/<step>/eval_info.json`; a one-liner aggregates the
success-rate-vs-step curve afterwards. ACT screen ≈ 5 ckpts × 10 eps × ~17 s ≈ 20 min
locally — cheap enough to run in the background while the replay engine is built.
Diffusion screen uses `--policy.num_inference_steps=10` (DDIM would need a scheduler
swap; fewer DDPM steps is the supported knob) only if the DDPM-100 cost proves painful.

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

### Build-phase tests (03_mock_deploy.py + 03b_screen_checkpoints.sh)

- [x] **T7 PASS (after two fixes)** — quick diffusion replay (2 episodes, stride 10):
  first run over-ran the dataset (D9), fixed; then RMSE **23 px at depth 1 → 70 px at
  depth 32** — error grows with prediction depth exactly as designed.
- [x] **T8 PASS (after one fix)** — quick ACT replay (1 episode): first run crashed in
  ACT's forward (D10), fixed; then 20 states in 4 s. ACT inference is ~1000× cheaper
  than diffusion's (one transformer pass vs 100 UNet denoising passes) — an asymmetry
  worth a sentence in the report.
- [x] **T9 PASS — full ACT replay:** all 2,500 held-out states (5 eps × 500, stride 1)
  in 150 s. Overall joint-space RMSE **0.097 rad**; depth curve **0.056 → 0.111 rad**
  (d=1 → d=100); per-episode MSE tight (0.0055–0.0113) → no outlier episode.
- [x] **T10 PASS — ACT checkpoint screen (D8 fixed on the way): resolves D7.**
  10 episodes/checkpoint, seed 42:

  | checkpoint | success | avg max reward |
  |---|---|---|
  | 20K | **20%** | 2.3 |
  | 40K | 10% | 1.3 |
  | 60K | 10% | 2.1 |
  | 80K | 10% | 2.0 |
  | 100K (final) | **0%** | 1.4 |

  The final checkpoint is the *worst* closed-loop while being the best open-loop — the
  robomimic result reproduced on our own run (overfitting to demonstrator idiosyncrasies
  hurts rollout before it hurts prediction MSE). 20% at 20K is in the ballpark of the
  original ACT paper's ~20% on human-demo insertion. **Decision: ACT's deployed
  checkpoint = 20K**, pending the 50-episode confirm run.
- [ ] **T11 — full diffusion replay** (21 eps, stride 4, K=8 probe) — running,
  ~2 min/episode after two false starts (D11, D12).
- [x] **T12 PASS — ACT 20K confirm run:** **20% success over 50 episodes**
  (avg max reward 2.26, seed 42) — the 10-episode screen estimate held exactly.
  This is ACT's headline closed-loop number; deployed checkpoint = **20K**.
- [ ] **T13 — diffusion checkpoint screen** (8 ckpts × 10 eps) — queued after T11.

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
- **D7 (T6 → RESOLVED by T10):** ACT scored **0.0 reward in both** ALOHA smoke episodes
  despite purposeful, coordinated arm motion (video evidence in `reports/m4_smoke/`).
  Candidate explanations were (a) closed-loop compounding error, (b) final-checkpoint
  overfitting per robomimic, (c) env-side mismatch. **The screen confirmed (b):** the
  20K checkpoint succeeds 20% of the time while the final 100K checkpoint succeeds 0%
  — the smoke test just happened to use the worst checkpoint. Success decays
  monotonically-ish with training steps while open-loop RMSE improves: our own
  open-loop/closed-loop divergence, on our own data (§4 T10).
- **D8 (03b, tooling):** the current `hf download` CLI prints `path=<dir>` — with a
  literal `path=` prefix — instead of the bare directory. Captured verbatim into
  `--policy.path`, it produced an `HFValidationError` two layers deep in draccus.
  Fix: `${SNAP#path=}` strip in the runner.
- **D9 (T7, dataset API gotcha):** `LeRobotDataset(episodes=[190, 191])` filters the
  *frames*, but `meta.episodes` still returns the **full** episode table with **global**
  `dataset_from_index/to_index` — naively iterating it walks episodes 0, 1, 2… and runs
  off the end of the subset (`IndexError: 369 out of bounds for size 363`). Fix:
  filter the table to the selected episodes and rebuild subset-local offsets from
  episode lengths (asserting they sum to `len(ds)`).
- **D10 (T8, policy I/O asymmetry):** requesting a delta-timestamps window for obs keys
  when the policy defines none (ACT: `observation_delta_indices=None`) adds a time dim —
  `observation.state` becomes `(B, 1, 14)` — and ACT's forward crashes with a token
  stack-size mismatch. Diffusion *requires* the window; ACT *rejects* it. Fix: only
  window obs keys when `observation_delta_indices is not None`.
- **D11 (T11, MPS memory ceiling):** batch 64 × DDPM-100 on the 262M UNet blew past
  unified memory — the process sat in uninterruptible waits with a ~417 GB virtual
  mapping and ~5 min of CPU per 24 min of wall time, thrashing instead of computing
  (system swap hit 5.9/7 GB). Batch 16 is the validated ceiling for this model on this
  machine. Batch size is not just a throughput knob at inference time either.
- **D12 (T11, observability — killed a healthy run):** Python **block-buffers stdout
  when redirected to a file**, so a long background run can compute for 40 minutes while
  its log shows nothing — indistinguishable from a hang. Diagnosed by re-running the
  known-good quick config (full speed) → the "stuck" full run had almost certainly been
  fine; its progress lines were sitting in the buffer when it was killed. Rule adopted:
  **always `python -u` for long redirected jobs**, and treat "no log lines" as
  unproven-hang until CPU-time accounting says otherwise.

## 6. Interfaces to Stage 4 (visualization)

Stage 4 consumes only files produced here — it never re-runs inference:

```
outputs/mock_deploy/<policy>/
├── ep<id>.npz            # ts, pred [T,H,A], gt [T,H,A], pad [T,H],
│                         # multi_ts [Tm], multi_pred [Tm,K,H,A]
└── metrics.json          # mse_by_depth, overall RMSE, per-episode stats, config echo
outputs/eval/screen_<policy>/<step>/   # checkpoint screen: eval_info.json + videos
outputs/eval/confirm_<policy>/<step>/  # 50-episode confirm on the winning checkpoint
```

## 7. Papers

- DAgger — Ross et al., 2011, arXiv:1011.0686 (compounding errors / covariate shift).
- "Tune to Learn" — arXiv:2604.02523 (open-loop MSE can rank policies opposite to
  closed-loop success; the reason we report both).
- Implicit BC — Florence et al., 2021, arXiv:2109.00137 (PushT origin + multimodality).
- robomimic — arXiv:2108.03298 (checkpoint selection by rollout, the rule part (b)
  implements).
