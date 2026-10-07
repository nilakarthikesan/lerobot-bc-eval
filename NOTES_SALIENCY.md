# Occlusion Sensitivity Notes — Stage 6 / M7

The first stage that asks **why** a checkpoint fails rather than **whether** it does. Same
discipline as the other stage notes: design before code, every decision with a reason, a
verification protocol run before bulk generation (§4), and a live issues log (§6) feeding
the report.

**The rule inherited from the pipeline design:** the probe (`scripts/08_saliency.py`) is
the only thing here that runs inference. It writes arrays to `outputs/saliency/*`; the
figures (`scripts/09_saliency_figures.py`) consume only those arrays plus dataset frames
and the existing `outputs/eval/screen_*` results — they **never re-run inference**.

---

## 0. The claim under test

M4 established the project's headline: ACT at 100K steps imitates the demonstrator better
than at 20K, yet succeeds **0%** in closed loop vs **20%** at 20K. The only way we could
tell the good checkpoint from the bad one was to run 50 simulator rollouts.

**Hypothesis:** the vision encoder's *attention pattern* degrades measurably before (or
alongside) task performance, so a laptop-speed probe that never touches the simulator can
flag a bad checkpoint.

Ground truth to grade against (already on disk, `outputs/eval/screen_*`):

| Policy | Closed-loop success by step | Reliable contrast |
|---|---|---|
| ACT | 20K=20%, 40K=10%, 60K=10%, 80K=10%, 100K=0% | 20K vs 100K (20K also 20% over 50 eps in `confirm_act`) |
| Diffusion | 25K=20%, 50K=0%, 75K=30%, 100K=20%, 125K=40%, 150K=20%, 175K=30%, 200K=50% | monotone-ish, no collapse — the **control group** |

**A null result is still a finding.** If attention is identical across ACT checkpoints,
the failure lives in the action decoder, not the eyes — which usefully narrows the
diagnosis and is reported as such.

## 1. Method: occlusion sensitivity

Slide a grey patch over the input image; at each position re-query the policy and measure
how far the predicted action chunk moves from the unoccluded baseline. Large movement =
the policy depended on what was under the patch. This is the same slide-and-score idea as
a convolution, lifted one level: the "filter" is a grey square and the "score" is *how
much the policy cares*. (Zeiler & Fergus 2014, occlusion maps.)

```
saliency[r, c] = mean_over_chunk( | policy(image with patch at r,c) - policy(image) | )
```

Grey = the per-channel dataset mean after normalization (i.e. 0 in normalized space), so
the patch is the least-informative input rather than a black hole that itself screams
"edge."

## 2. Input data contract (verified against the actual configs)

```
ACT  : observation.images.top  [3, 480, 640], single obs step (n_obs_steps=1)
       observation.state       [14];  chunk H=100, A=14 (joint radians)
Diff : observation.image       [3, 96, 96], 2-step window (n_obs_steps=2)
       observation.state       [2];   chunk H=32, A=2 (xy workspace)
```

Two consequences for the code:

- **ACT** has `observation_delta_indices=None`, so obs keys are not windowed — the image
  tensor is `[B, 3, 480, 640]`. Occlude the single frame.
- **Diffusion** windows obs to `[B, 2, 3, 96, 96]`. Occlude the **same spatial patch in
  both timesteps** so the encoder sees a consistent occlusion (occluding one frame only
  would leak the answer via the other).

Frames for the heatmap backdrops come from `LeRobotDataset(episodes=[ep])` video decode —
data access, not inference.

## 3. Outputs and metrics

```
outputs/saliency/<policy>/step<N>_ep<id>.npz
  saliency   [F, gh, gw]   per-frame occlusion map on the patch grid
  frames_ts  [F]           frame index within the episode
  grid       (patch, stride, H, W)   echo for exact upsampling in figures
  meta       json string: policy, step, episode, image_key, n_positions
```

Two **label-free** metrics (no segmentation masks needed):

- **Focus** = peak / mean of the saliency map. "Locked onto something" vs "vaguely
  sensitive everywhere." Higher = more focused.
- **Motion overlap** = fraction of saliency mass inside the episode's *motion mask*
  (thresholded frame-to-frame pixel difference across the episode, resized to the grid).
  Proxy for "watching the task vs the furniture." Higher = attention on things that move.

Both are computed in the figure stage from the saved maps, so thresholds can be retuned
for free.

## 4. Verification protocol (before any bulk generation)

Mirrors the V-series discipline from M5. Each gates the next:

- **S1 (baseline stability):** same frame queried twice unoccluded → identical chunk for
  ACT (deterministic), near-identical for Diffusion (fix the seed; residual is the noise
  floor the signal must clear). If the baseline is noisy, the difference signal is
  meaningless.
- **S2 (extreme sanity):** occluding the *whole* image must produce a large action change.
  If it doesn't, the policy is ignoring vision and running on proprioception — itself the
  finding.
- **S3 (spatial sanity):** on a PushT frame the top-saliency cell should overlap the
  pusher or the T-block, not empty table. Checked by eye on 3 frames.
- **S4 (patch-size robustness):** one frame at two patch sizes; conclusions must not flip
  with the hyperparameter.
- **S5 (metric sanity):** focus/overlap must differ between a scrambled image and a real
  one.

### Results

- **S1 PASS.** ACT baseline queried twice is bit-identical (`max|diff| = 0.0`; it is
  deterministic — z=prior mean). Diffusion with fixed noise has a residual of 3.25 px in a
  ~512 px workspace = **0.9% of action scale** (MPS float non-determinism, not sampling —
  the noise tensor is held fixed). The occlusion signal must clear this ~3 px floor;
  observed peaks are far above it (peak/mean ≈ 6).
- **S2 PASS.** Occluding the whole image moves the predicted action by **57% of scale for
  ACT** and **21% for Diffusion** — both policies genuinely depend on vision, so the probe
  is measuring something real (not a proprioception-only controller).
- **S3 PASS (spatial sanity).** In `reports/m7/F3`, Diffusion 200K saliency sits squarely
  on the pusher (blue disc) and the T-block, not on empty table; in `F1` ACT saliency
  concentrates on the two arms / insertion zone. The probe attends to task objects.
- **S4 PASS (patch-size robustness).** The ACT smoke (patch 96 / stride 96) and the full
  run (patch 96 / stride 48) put the hot region in the same place; the conclusion does not
  flip with the grid density.
- **S5 PASS (metric sanity).** The motion mask covers 40% of cells, so a spatially random
  saliency map would score `motion_overlap ≈ 0.40`. Observed values separate cleanly from
  chance: the scattered Diffusion 25K map sits at **0.48** (just above chance, matching its
  visibly diffuse heatmap) while focused maps (ACT ≈ 0.76-0.79, Diffusion 200K = 0.76)
  sit at ~2× chance. The metric distinguishes "locked on" from "sensitive everywhere."

## 5. Honest limitations (stated up front, per the report's tone)

- Screen success uses n=10, so intermediate ACT checkpoint gaps are within noise. Only the
  **20K vs 100K endpoint contrast** is load-bearing; the middle points are a trend line.
- Occlusion attribution is crude — a grey patch is itself mildly out-of-distribution.
  Report it as a *signal*, not proof of causation.
- Five ACT checkpoints = five points. This is a proof of concept for a trust signal, not a
  validated predictor.

## 6. Issues log (live)

- **M7-1 (network hang, then offline hard-error):** `LeRobotDataset(root=None)` and
  `snapshot_download(...)` probe the Hub for refs on construction. On this machine (45-day
  uptime, flaky HF connectivity) that call **hangs indefinitely** — the probe sat at 0%
  CPU blocked on a socket read for >12 min. Setting `HF_HUB_OFFLINE=1` converts the hang
  into an immediate `OfflineModeIsEnabled` *error* instead, because the metadata loader's
  default path (`HF_LEROBOT_HOME/<repo>`) does not match where the data actually lives
  (the revision-safe snapshot cache under `.../lerobot/hub/datasets--*/snapshots/<hash>/`).
  **Fix:** force offline *and* pass `root=<globbed snapshot dir>` so `_load_metadata()`
  reads straight from disk with zero network round-trips (`dataset_root()` in the script).
- **M7-2 (cold-start / decode cost on a memory-starved box):** 16 GB RAM, ~300 MB free,
  3.5 GB swap in use → cold `import torch`/dyld mmap page-ins take ~90 s and the first
  video decode ~20 s/frame. Building the dataset and decoding frames **once per policy**
  (not per checkpoint) and reusing the decoded tensors across all checkpoints cuts the
  full 5-checkpoint ACT run from an estimated ~25 min to a few minutes once warm.
- **M7-3 (API asymmetry):** `DiffusionPolicy.predict_action_chunk(batch, noise=...)`
  accepts a noise arg (needed to hold sampling fixed for S1); `ACTPolicy` does not. The
  probe passes `noise=None` for ACT.

## 7. Results and findings

Metrics per checkpoint (`scripts/09_saliency_figures.py --metrics`; figures in
`reports/m7/`):

| Policy | step | success % | focus (peak/mean) | motion overlap |
|---|---|---|---|---|
| ACT | 20K | 20 | 5.8 | 0.761 |
| ACT | 40K | 10 | 6.0 | 0.773 |
| ACT | 60K | 10 | 6.4 | 0.780 |
| ACT | 80K | 10 | 6.4 | 0.791 |
| ACT | 100K | 0 | 6.4 | 0.790 |
| Diffusion | 25K | 20 | 3.7 | 0.481 |
| Diffusion | 100K | 20 | 5.9 | 0.576 |
| Diffusion | 200K | 50 | 10.0 | 0.757 |

Pearson r(success, ·): ACT focus **−0.76**, overlap **−0.82**; Diffusion focus/overlap
**+0.94**.

The maps become more concentrated and overlap the moving task region more across the sampled checkpoints. This does not track rollout success in the same direction for ACT and Diffusion.

The probe measures changes in the full policy's action output after an image perturbation. Calling it encoder attention is an interpretation, not a direct measurement. A concentrated map does not prove visual representations improved, and it does not locate failure downstream of the encoder. Controlled perception/decoder interventions and larger rollout sets are needed to test those hypotheses.

The five ACT and three Diffusion observations are exploratory. The checkpoint-screen success estimates use ten episodes, so the correlations should not be treated as validated trust signals or causal diagnoses. The maps can guide follow-up inspection of observation dependence, but cannot replace rollout evaluation.

## 8. References

- Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks*, ECCV 2014
  (occlusion sensitivity maps).
- robomimic, arXiv:2108.03298 (open-loop vs closed-loop divergence — the gap this probe
  tries to explain).
