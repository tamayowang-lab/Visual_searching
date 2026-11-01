# Phase 4B: spatial attention and active perception

The archived Phase 3 and Phase 4A numerical paths remain available by leaving
`active.enabled` false. The opt-in path logs every observation and decision. It
has no head, eye-plant, blink or robot controller. This phase asks whether the
spatial observer is useful before adding those layers.

## Components and experiments

- **A**: exact Phase 3 heuristic-lognormal spatial/timing comparator, with traces.
- **B**: explicit valid-stimulus mask. Never infer padding from darkness.
- **C**: search instruction versus free exploration; coarse current-view semantic
  cues can guide attention before central recognition. No ground-truth target box.
- **D**: category/area-normalized log-priority competition. Task guidance,
  supported information gain, movement cost and inspection compete in declared
  dimensionless units. This is a design model, not a fitted neural mechanism.
- **E**: retain valid fixed anchors and add up to two local support peaks per
  region/object. Split each parent's valid-area quota across its proposals.
- **F**: bounded multiscale peripheral sensor and persistent tentative proto-
  objects that can acquire central evidence, merge, decay or be discarded.
- **S**: E plus the bounded sensor, without proto-objects.
- **N**: F without across-fixation proto-object persistence.

All components have separate `active` configuration fields. `variant` is a
report label; the flags determine behavior. `configs/phase4b-*.yaml` specify the
exact comparison. F changes both sensor and representation; S and N isolate
those effects. `configs/phase4b-explore.yaml` is the simpler D exploration demo.
D is a provisional comparator selected from validation, not a software default
replacement or a statistically established winner. Existing timing variants
`heuristic_lognormal`, `accumulator`, `accumulator_hold` remain separately usable.

Peripheral semantic inference receives only downsampled current foveated RGB
(C–E), or an eccentricity-dependent 160/80-wide bounded RGB sensor (F/S/N).
Neither uses a full-resolution whole-scene semantic bank. Raw detector scores
are thresholded at .15 then capped/weighted by .35 and sensory reliability;
these are **uncalibrated evidence**, not probabilities of correct recognition.
The central detector retains the existing gaze gate. F's recognized status
requires central semantic evidence and accumulated central exposure. Relative
depth is the existing monocular estimate, committed locally where observed.

D combines salience (.8), target evidence (3), useful information (uncertainty
reduction × sensory support × novelty), movement (-.5 × normalized distance² /
.20²), inspection (-.35 × history × (1−task)) and existing near-gaze persistence.
Task/object/region/stay groups have declared priors .55/.15/.25/.05 in search;
object/region/stay priors are .45/.50/.05 in exploration. Missing groups are
renormalized. Log group prior and log within-group area share enter each
candidate's complete utility before a temperature-1 softmax. These assumptions
are explicit, not tuned to the old six test scenes. Candidate multiplicity
cannot by itself increase a group's base mass. More proposals can still alter
its competition through different evidence and movement costs.

## Run locally

Install the project and verified CPU vision extra, then run the existing model
setup script as documented in the main README. Config checkpoint paths are
relative to the project. A classical provider is also usable for mechanism
inspection but supplies no semantic target recognition.

```powershell
active-observer run-one your-photo.png --config configs/phase4b-D.yaml --target chair --out outputs/phase4b
active-observer run-one your-photo.png --config configs/phase4b-explore.yaml --out outputs/explore
```

For a padded photograph, explicitly pass `--stimulus-bounds X1 Y1 X2 Y2`, or
`--valid-mask mask.png` / `mask.npy` / `mask.npz` (key `mask`). Coordinates refer
to the supplied image. All valid-stimulus policies reject invalid destinations,
assign rejected candidates zero probability, mask priority displays and reject
invalid starting gaze. Genuine dark photograph content stays valid. Without
an explicit mask/bounds, the entire supplied photograph is valid.

Outputs: `run.json`, `decision-traces.jsonl`, `dashboard.png`, `inspector.html`.
The HTML is standalone UTF-8; no server or CDN is needed. It displays the new
mechanism inspector for opt-in runs. Batch runs also export traces and accept
`--target`. No user-uploaded photo is sent to a remote inference service.

## Decision trace contract

Every fixation records actual current RGB hash and geometry, perceived central
detections, current coarse evidence/feature maps, proto hypotheses/events,
memory snapshots before/after the observation and per-object changes, available
and rejected candidates, all utility terms, normalized probabilities and ranks.
The outgoing choice is separately identified from the incoming prior choice.
The RNG uniform draw and pre-choice RNG state permit replay. Timing includes
onset candidates and pre-timing RNG state. RNG states are JSON **strings** to
avoid JavaScript losing 64/128-bit integer precision.

For softmax selection the recorded probability is the actual choice mass. For
optional readiness-race selection it is null; the diagnostic softmax mass is
separate. A sampled destination can fail to arrive before the viewing window;
`arrived` distinguishes that case. Terminal observations log an explicit stop.
Memory changes describe observation updates; pre-observation snapshots already
include any earlier intersaccadic decay. They are not fabricated verbal reasons.
The priority raster is a smoothed display of the exact probability vector,
not a second hidden sampling policy. Display thumbnails are reconstructed and
checked against recorded full-input SHA256 hashes before being encoded. Rebuild
frames with the same pinned NumPy/OpenCV environment as the producer; a hash
mismatch raises an error rather than displaying an approximate substitute.
The hosted replay omits the optional dense depth-reference field (the original
result stays local); all fixation decisions and complete traces are preserved.
Hosted replay files are gzip-compressed; the browser decodes them locally.

The browser replays actual Python outputs. Candidate clicks inspect components;
they do not launch inference. The original scene explorer's click-start demos
remain available. Human-prefix next coordinates are evaluation-only labels,
read only after candidates and choice are frozen. Prefix records condition on
human gaze history and are clearly distinct from autonomous model scanpaths.

## Evaluation and reproduction

`prepare_phase4b.py` prepares a **private** 12-scene independent-to-development
subset from licensed local originals plus the released COCO-Search18 fixation
JSON. The image order is SHA256(`phase4b-independent-v1:filename`), excluding
all 24 earlier scenes; first six validation, next six test. The source mirror
is pinned to `401baae571e325eacb000f21783463d2343609ab`. No scene is selected by
performance. This is released training material, not the official benchmark
holdout; COCO-pretrained vision weights may have seen these photographs.

```bash
python scripts/prepare_phase4b.py source-list.json previous-manifest.json raw-fixations.json originals private-new-scenes
python scripts/evaluate_phase4b.py --manifest private-new-scenes/manifest.json --config configs/pretrained-cpu.yaml --fit reports/phase3/phase3-fit.json --raw raw-fixations.json --variant D --split test --seeds 11 23 37 --out private-evaluation/test-D
python scripts/prefix_phase4b.py --manifest private-new-scenes/manifest.json --config configs/pretrained-cpu.yaml --fit reports/phase3/phase3-fit.json --raw raw-fixations.json --variant D --out private-evaluation/prefix-D
```

Repeat for A–F/S/N. Corrected dynamic outputs in this run are named
`test-F-revision2`, `test-N-revision2`, likewise validation and prefix folders.
Run `report_phase4b.py --work private-evaluation --manifest ... --raw ... --out
reports/phase4b` for aggregate statistics, and `build_phase4b_lab.py` for the
internal research website. All CLI scripts expose `--help`.

Frozen original lognormal timing fit; seeds 11/23/37; human initial gaze and
RT-conditioned exposure; same photos, tasks, providers and coordinate transform.
Seeds/steps are averaged within participant/task, tasks within participant,
participants within scene; scenes are primary units. Crossed scene/participant
bootstrap intervals and scene sign-flip diagnostics are exploratory at n=6.
Never pool simulation repetitions as independent scenes. Count and dwell
heatmaps, center-adjusted log-density gain, approximate visual-angle candidate
coverage, probability allocation, distributions and target performance are
reported separately. There is no arbitrary combined realism score.

Cumulative RT clipping ignores unavailable saccade gaps; this is a limitation,
not response-time prediction. Conditional timing excludes initial/terminal
fixations. Survival is only a terminal-censor sensitivity. Correct target-
present trials cannot validate absent-target stopping or free exploration.
Target-box detection precision, target-location Brier and internal memory
retention are limited proxies. Full object accuracy, semantic calibration and
human memory require new labels/tests; do not claim they were established.

A trace/source audit discovered inherited attributes in new proto candidates
while test jobs had started but **before any held-out outcome scores were
inspected**. Revision2 computes proto attributes from their own location and
confidence. F/N were revalidated and rescored; unaffected A–E/S trajectories are unchanged; A metadata is regenerated with declared
bounds for honest baseline padding display, while B–E/S caches retain
their producer hashes. Invalidated F/N engineering results are excluded.
`phase4b-protocol.json` records the amendment. The test is described as
exploratory independent-scene evidence, not preregistered confirmation.

## Data handling

Public GitHub contains code, configurations, tests, protocol, aggregate plots
and statistics only. Raw photographs, individual human trajectories and full
empirical decision replays stay local or in the existing **owner-only internal
research Site**. Never copy the generated empirical lab directory into a public
repository or change its access scope. The Porto demonstration uses the existing
shareable scene and has no empirical ground truth. Other scenes/tasks and
independent free-viewing data are needed for generalization claims.
