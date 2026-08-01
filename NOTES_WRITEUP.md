# Writeup Notes — Stage 5 / M6 (the report)

The last stage. Unlike M2–M5, **nothing new is computed here** — the report assembles
material that already exists in the stage notes, `reports/`, and `outputs/`. The design
work is therefore editorial: what to claim, what evidence carries each claim, and what
to leave out. Same discipline as the other stages: design first (this doc), then build
(`reports/REPORT.md`), then verify (§4), then ship.

---

## 0. Deliverable & audience

- **Artifact:** `reports/REPORT.md` — the document Irmak reads, per DESIGN.md §8,
  alongside the public repo and the Viser demo. Explicit requirement from the task:
  include **the LeRobot issues we encountered**.
- **Audience:** robotics-literate reader who has NOT followed our notes. Every term we
  coined (D7, "screen/confirm", anchor states) must be introduced or avoided.
- **Length target:** ~4 pages of prose + figures. The notes stay the deep archive;
  the report is the tour.

## 1. The two lead findings (the report's spine)

Everything else supports these:

1. **Checkpoint selection by rollout is not optional.** ACT's best closed-loop
   checkpoint is its *earliest* (20K: 20% success; final 100K: 0%) while diffusion's
   is its *final* (200K: 48%). Same pipeline, opposite outcomes — you cannot know
   without rolling out (robomimic's thesis, reproduced end-to-end on our own runs).
2. **Open-loop fidelity ≠ closed-loop competence.** ACT's open-loop RMSE kept
   improving with training while its rollout success collapsed — our own instance of
   the covariate-shift gap the literature warns about (DAgger; arXiv:2604.02523).

Supporting findings, one paragraph each: the tight-fan/near-determinism result with
Soare's independent corroboration (V-A); the event-anchored difficulty structure in
the F6 heatmap ("approach is easy, contact is hard" — the open-loop echo of D7); the
train/deploy version-skew incident (D2) as the flagship MLOps lesson.

## 2. Report outline (claims → evidence map)

| § | Content | Evidence it embeds / links |
|---|---|---|
| 1 Task & approach | one paragraph + pipeline diagram | diagram from NOTES_PIPELINE.md |
| 2 Policies & data | why ACT + Diffusion over naive BC; datasets + splits | NOTES.md stage 2; EDA figs (`reports/eda/`) |
| 3 Training | resolved configs table, loss curves, HF Jobs A100 runs | training/ docs; Hub model links |
| 4 Mock deployment | the M4 verdict table; screen curves; confirm numbers | NOTES_DEPLOYMENT.md §4; `outputs/eval/*/eval_info.json` |
| 5 Findings | the two lead findings + supporting three (§1 above) | depth_curves.png, screen tables, fans, heatmap |
| 6 Visualizations | F2 stills, mp4, Viser scene (screenshots + run command) | `reports/m5/`, scripts/05 |
| 7 Issues log (required) | curated D1–D12 + V-A–V-E + training-stage issues, grouped by theme | all stage notes |
| 8 Limitations & next | 185-vs-206-episode gap vs official 65%; backlog items | NOTES_VISUALIZATION.md §7 |

**Figure budget (6):** depth curves; PushT six-frame overlay; top-spread fans; ALOHA
error heatmap; one Viser screenshot; ACT screen success-vs-checkpoint table rendered
as a small chart. Everything else is linked, not embedded.

**Issue curation rule:** group by theme, not chronology — (a) version skew & packaging
(D2, D3, D4, V-C), (b) silent-failure traps (D6, D9, D10, D12, V-D), (c) resource
ceilings (D11, HF-Jobs gym gap D1), (d) evaluation gotchas (D5, D7/D8). Each issue:
one sentence of symptom, one of root cause, one of fix. The full forensic detail stays
in the stage notes, linked.

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| Format | Markdown in-repo (`reports/REPORT.md`) | renders on GitHub where the repo is reviewed; no build step |
| Numbers policy | every number traced to a JSON/log file (§4 checklist) | a report with unverifiable numbers is a liability |
| Tone on the 48%-vs-65% gap | state it plainly with the three attributable causes | honesty is the point of the issues log |
| What to leave out | per-hyperparameter rationale, shell mechanics, EDA minutiae | linked notes cover them; the report is 4 pages |

## 4. Verification protocol (a document gets tested too)

- [ ] **W1 — number audit:** every metric in the report grep-matches its source
  (`metrics.json`, `eval_info.json`, screen logs). No number from memory.
- [ ] **W2 — link audit:** every relative link and image path resolves on GitHub
  (render check on the pushed commit).
- [ ] **W3 — reproduce audit:** the README quickstart + script commands named in the
  report actually exist and match current CLI flags.
- [ ] **W4 — cold-reader pass:** no undefined jargon from our internal shorthand.

## 5. Definition of done (the finish line)

1. `REPORT.md` written per §2, W1–W4 pass, pushed.
2. README links the report at the top; M6 checked off.
3. Optional (backlog, not blocking): K≈100 fan probe, Rerun inspector, PushT-in-Viser.
4. Send Irmak the repo link + report link + Viser run command.
