# Visualization Notes — Stage 4 / M5

The in-depth design for turning M4's raw arrays into the figures and demo artifacts the
report leads with. Same discipline as the other stage notes: design before code, every
decision with a reason, and a live issues log (§6) feeding the M6 writeup.

**The rule inherited from the pipeline design:** Stage 4 consumes only files M4 produced
(`outputs/mock_deploy/*`, `outputs/eval/*`) plus dataset video frames — it **never
re-runs policy inference**. Iterating on plots must stay free.

---

## 0. What each figure has to argue

A figure earns its place by carrying one specific claim from the M4 results:

| Claim (from NOTES_DEPLOYMENT.md §4 verdict) | Figure |
|---|---|
| Prediction error compounds with depth — for both policies | F1 depth curves |
| The policies actually imitate: predictions track the demonstrator in the workspace | F2 PushT overlay stills / F5 ALOHA joint panels |
| Diffusion captures **multimodality** — the reason it exists (Chi et al. 2023, Florence et al. 2021) | F3 fan plots |
| The whole behavior, legibly, for a non-reader of tables | F4 episode animation |

## 1. Input data contract (verified against the actual npz files)

```
outputs/mock_deploy/diffusion_pusht/ep<id>.npz     # 21 episodes, stride 4
  ts [T]              frame index within episode (0, 4, 8, …)
  pred [T, 32, 2]     predicted chunk, workspace coords (x, y) ∈ ~[0, 512]
  gt   [T, 32, 2]     demonstrator actions, same frame alignment (chunk d = t+d)
  pad  [T, 32]        True where GT is episode-end padding → mask from metrics/plots
  multi_ts [Tm], multi_pred [Tm, 8, 32, 2]   K=8 fresh-noise samples at probe states

outputs/mock_deploy/act_aloha/ep<id>.npz           # 5 episodes, stride 1
  same schema with H=100, A=14 (joint radians), K=1
metrics.json per policy: mse_by_depth, per-episode stats  (F1 reads only this)
```

Frames for overlays come from `LeRobotDataset(episodes=[ep])` video decode — data
access, not inference.

## 2. The coordinate mapping (the stage's one real risk)

PushT actions are end-effector targets in the **512×512 pymunk workspace**; the stored
observation is a **96×96 render** of it. Every overlay stands or falls on the transform

```
pixel = action * (96 / 512)          # scale — and possibly a y-flip, TBD empirically
```

Whether gym-pusht's render flips y is not worth trusting documentation for. **The
frame-by-frame test (§4, V1) settles it visually:** overlay the GT *action* trace on
consecutive decoded frames — the demonstrator's actions are cursor positions, so the
trace must track the blue pusher circle as it moves. If the trace mirrors the pusher's
motion vertically, we flip; if it tracks, we ship. The verified mapping gets recorded
here either way.

ALOHA needs no mapping for the baseline figures: actions are 14 joint angles plotted in
their own axes. (FK to 3D EE space via MuJoCo is the stretch, §5.)

## 3. Figure inventory (files under `reports/m5/`)

- **F1 `depth_curves.png`** — RMSE(d) for both policies, side-by-side panels (units
  differ: px vs rad — never share a y-axis). Marks the headline numbers: 18.5→65.2 px
  over 32 steps; 0.056→0.111 rad over 100 steps. Padding already excluded in M4.
- **F2 `pusht_ep<id>_frames.png`** — a row of ~6 stills spanning one episode: decoded
  frame + GT future (green) + predicted chunk (red), fading with depth. The
  frame-by-frame verification artifact *and* a report figure.
- **F3 `pusht_fans_ep<id>.png`** — at probe states: 8 translucent sample trajectories +
  GT. The money shot: ambiguous states (e.g. pusher between two approach directions)
  should show visibly forked fans; confident states, a tight bundle. If no state forks,
  that is a *finding* (mode collapse?), not a plotting failure.
- **F4 `pusht_ep<id>.mp4`** — the F2 overlay animated over every queried state
  (matplotlib frames → ffmpeg). One episode is enough for the demo.
- **F5 `aloha_ep<id>_joints.png`** — 14 small panels (7×2), one per joint: GT episode
  trace (thin, full length) + predicted 100-step chunks branching off at ~5 anchor
  states (colored by anchor). Shows where prediction hugs truth vs drifts — the
  joint-space equivalent of F2.
- **F6 `aloha_ep<id>_error_heatmap.png`** — |pred − gt| averaged per (anchor state ×
  depth), joints aggregated: shows *when* in the episode ACT is hardest to predict
  (expect: contact/grasp phases, per the D7 story).

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Library | matplotlib only for report figures | zero new deps; Viser is the separate interactive artifact |
| Which episodes | fixed defaults (pusht 190, aloha 47) + `--episodes` flag | reproducible figures; 190/47 already used in sanity tests |
| Chunk overlay density in F2/F4 | every queried state for F4; ~6 evenly-spaced for F2 | animation wants continuity, stills want legibility |
| Depth fade | alpha ramp along the 32/100 steps | encodes "farther future = less certain" without a legend |
| Fan colors | one hue per sample, α≈0.6 | forks must be countable |
| Y-axis truth | decided by V1 frame test, then hard-coded + documented | see §2 |

## 4. Frame-by-frame verification protocol (before any full generation)

Run on single frames, inspect the PNGs visually, in this order — each test gates the
next:

- **V1 (coordinate truth):** one PushT frame + GT trace overlay at 3 timesteps spread
  across episode 190. PASS = green trace sits on/ahead of the blue pusher circle in all
  three, tracking its motion between frames. FAIL modes: mirrored (flip y), offset
  (origin bug), wrong scale (trace exceeds frame).
- **V2 (alignment in time):** at state t, the GT trace's *first* point must coincide
  with the pusher's current position (chunk step d=0 == action at t — M4's verified
  alignment, now checked visually).
- **V3 (prediction sanity):** predicted chunk (red) at the same 3 states — must start
  near the pusher and head somewhere task-plausible (toward the T / target zone), per
  the T5 eval videos where the policy demonstrably works.
- **V4 (fan sanity):** one probe state's 8 samples — spread should be modest at
  mid-push (committed motion) and, if any state forks, it should be an approach state.
- **V5 (ALOHA panels):** joint 0 & 6 (left arm shoulder + gripper): predicted chunks
  must continue the GT curve's local trend from each anchor, not jump.

Only after V1–V5 pass: full generation for all default episodes + review + commit.

### Results (all passed; see `reports/m5/`)

- **V1 PASS — no y-flip.** `pixel = action × 96/512`, verified across 3 frames of
  ep 190: the GT trace tracks the blue pusher and the grey T converges on the green
  goal region. The verified mapping is hard-coded as `PUSHT_SCALE` with a comment.
- **V2 PASS** with one honest nuance: at episode start the GT trace begins slightly
  *ahead* of the pusher — PushT actions are cursor **targets**, and the demonstrator's
  cursor leads the pusher before contact. Not an alignment bug (mid-episode states
  start exactly on the pusher).
- **V3 PASS** — predicted chunks start beside the pusher and wrap the T plausibly;
  tight agreement with GT at committed states (t=40, 120), legitimate divergence at
  ambiguous ones (t=0, 160).
- **V4 PASS as a finding** (see issues log V-A) — fans do not fork; figure redesigned.
- **V5 PASS** — all 14 joint panels: chunks continue the GT trend from every anchor;
  interesting divergences around the grasp phase (L.wrist_r, L.forearm ~frames
  150–250).
- **Bonus finding from F6:** the error heatmap's bright streaks run **diagonally**
  (constant t+d), i.e. specific *events* — the grasp/contact moments at frames
  ~50–250 — are hard to predict from any anchor before them, while the post-transfer
  phase is near-zero error. Independent open-loop confirmation of the D7 story
  ("approach is easy, contact is hard").

## 5. Interactive artifact — IMPLEMENTED (`scripts/05_viser_aloha.py`)

The DESIGN.md §8 Viser scene, built on MuJoCo FK:

- **FK:** `gym-aloha`'s `bimanual_viperx_insertion.xml`; action layout
  `[left 6, left grip, right 6, right grip]` maps to qpos `0–5` / `8–13` (fingers
  `6,7`/`14,15` stay 0 — gripper opening doesn't move the gripper base). Read
  `vx300s_{left,right}/gripper_link` body positions after `mj_forward`.
- **Scene:** GT EE traces as green (left) / blue (right) splines; ACT's predicted
  100-step chunks as warm-colored splines branching at 5 anchor states; timestep
  slider drives two markers along the truth; description panel in-scene.
- **Run:** `python scripts/05_viser_aloha.py --episode 47` → link printed
  (auto-picks a free port if 8080 is taken).

### Verification (V6–V8, all passed)

- **V6 (FK sanity):** EE z ∈ [0.132, 0.307] m — plausible tabletop workspace — and
  max per-frame EE step 0.7 cm at 50 fps (smooth, no jumps/NaN).
- **V7 (render, via browser screenshot):** both arms' GT traces render with predicted
  chunks visibly hugging them; markers sit at t=0.
- **V8 (interaction):** setting timestep 0 → 400 moved both markers along their
  traces over the live websocket.

## 6. Issues log (live)

- **V-A (F3, finding not bug):** the 8-sample fans **collapse to a tight bundle** at
  essentially every probe state — median across-sample spread 1.6 px (max 7.7 px) in
  the 512-px workspace. Verified numerically that samples are *not* identical (fresh
  noise per call works); the trained policy is simply near-deterministic given
  context. Interpretation: the textbook forked-fan figures (DP paper, IBC) use
  deliberately symmetric/ambiguous initial states, while our probe states are
  mid-demonstration where the human had already committed; the policy earns its 48%
  by being decisive. **Fix to the figure, not the model:** F3 now ranks all probe
  states across all 21 episodes by spread and shows the top 4, with the spread
  quantified in each panel title (`pusht_fans_top_spread.png`). The largest spreads
  are at episode-start/approach states, consistent with the theory.
  **Independent corroboration (found post-hoc):** Alexander Soare's PushT
  [multimodality experiments](https://github.com/alexander-soare/little_experiments/blob/main/action_multimodality.md)
  reach the same conclusion with ~100 Monte-Carlo samples per state — "these models
  do not produce multi-modal action trajectory distributions; the distributions are
  much sharper than one might expect" — and show that (a) conditioning on the
  2-observation history (velocity) sharpens the distribution further, and (b)
  multimodality re-emerges if small Gaussian noise is added to the *observations*.
  Our K=8 result is the same phenomenon at smaller K.
- **V-B (F2, perception nuance):** GT trace leads the pusher at episode start
  (actions are cursor targets) — documented in V2 so nobody "fixes" it later.
- **V-C (05, API rot):** `matplotlib.cm.get_cmap` is gone in matplotlib ≥3.9 —
  crashed the first Viser launch. Use `matplotlib.colormaps["name"]`.
- **V-D (05, silent port fallback):** ports 8080/8081 were already occupied on this
  machine, and **Viser silently binds the next free port** — the script's "ready at
  {requested port}" message would have printed a dead link. Fix: report
  `server.get_port()`, never the requested port.
- **V-E (05, framing):** Viser's default camera fits the 1.2 m table grid, which makes
  the ~30 cm EE traces nearly invisible (caught by the V7 screenshot). Fix: set the
  camera on `on_client_connect` to frame the traces' centroid.

## 7. Improvement backlog (researched, not yet built)

Ranked by payoff-per-effort; none block M6:

1. **Monte-Carlo fans at K≈100** (Soare's protocol) at the top-spread states only —
   with 10 denoising steps this is ~10 min of compute and makes the "sharp
   distribution" claim quantitative (histogram of sample spread), not anecdotal.
2. **Observation-noise probe:** re-run the K-sample probe with small Gaussian noise on
   `observation.state` to *elicit* the latent multimodality (Soare shows it
   reappears). One extra CLI flag on `03_mock_deploy.py`.
3. **Temporal color gradient** on predicted trajectories (red→blue along the chunk,
   as in Soare's GIFs) instead of alpha fade — encodes flow direction legibly in
   stills.
4. **Viser playback button** (auto-advance the timestep slider) + per-anchor
   visibility toggles; optionally draw the predicted chunk *nearest* the current
   timestep only.
5. **Rerun-based episode inspector:** LeRobot ships `lerobot-dataset-viz` built on
   [Rerun](https://github.com/rerun-io/rerun), the community standard for exactly
   this (camera + joints + action chunks on one synced timeline). Our npz arrays
   could be logged as `LineStrips3D` alongside camera frames — strongest candidate
   if the writeup demo needs to scrub video and predictions together.
6. **PushT in the 3D scene:** the 2D workspace as a plane with the mp4 texture and
   prediction fans as flat splines — unifies both policies in one demo artifact.

## 8. References

- Diffusion Policy — Chi et al., RSS 2023, arXiv:2303.04137 (fan/multimodality framing).
- Implicit BC — Florence et al., 2021, arXiv:2109.00137 (PushT multimodality origin).
- robomimic — arXiv:2108.03298 (why we show closed-loop alongside open-loop figures).
