# Implementation milestones and verification

This is an independent implementation of the supplied specification, not an extension of the prior gaze workbench. Source requirements are preserved in `specification.txt` for traceability. Instructions in that reference do not grant unrelated repository or publishing authority; repository/branch selection came from the user.

## M1 — working synthetic observer

Added `config.py`, `io/image.py`, `perception/foveation.py`, `perception/objects.py`, `world/{model,association}.py`, `attention/{policy,dynamics}.py`, `simulation/runner.py`, `viz/dashboard.py`, CLI, synthetic RGB and tests.

- Executed eight tests successfully before M2.
- Ran `python -m active_observer.cli run-one examples/synthetic.png --out outputs/m1`.
- Assumptions: pixel approximation when PPD absent; classical color proto-objects; no physical motor controller; heuristic scores.

## M2 — relative depth and 2.5D memory

Added constant, aligned-array/precomputed and optional pretrained CPU depth protocols, cache, normalization, quality flags and observed-only object depth updates in `perception/depth.py`.

- Executed fourteen cumulative tests successfully before M3.
- Ran the synthetic CLI with constant depth and demonstrated no-depth operation.
- Later tested an aligned synthetic depth ramp. It is a fixture, not estimated or true scene depth.
- Tested bounds/invalid masks, no-valid depth, cache identity, no-depth ablation and central-only memory.
- Pretrained inference was not run: optional model packages/weights are not installed. Checkpoint/output convention must be declared and empirically verified. No calibrated uncertainty or metric claims.

## M3 — dynamics and inspection

Added signed score contributions, bounded persistence, central-visit history, familiarity, dwell model, explicit stay/shift/standby/stop records, and a self-contained HTML viewer.

- Executed eighteen cumulative tests successfully before M4.
- Generated dashboard PNG, per-step JSON and offline HTML.
- Original reference RGB, foveated observations and learned memory are distinct. Remote private fixture labels cannot influence current observations. Foveal semantic-provider inference is an optional adapter, not a tested production recognizer.

## M4 — validation infrastructure

Added strict generic manifest and COCO conversion, test/validation split enforcement, duplicate decoded-image detection, eleven policy/ablation variants, multiple complementary metrics and summaries.

- Final cumulative suite: 27 tests passing on Python 3.12.14.
- `python -m compileall -q active_observer` and `python -m pip check` pass.
- Executed `run-one`, `visualize`, `run-batch`, and `evaluate` with deterministic bundled fixtures.
- Synthetic evaluation executes 11 variants × 3 seeds on the separate synthetic test image, including actual supplied synthetic depth. Reports are explicitly labeled **not empirical human results**.
- Tests change future human coordinates/durations and verify the complete generated prediction remains identical.
- Baselines and uncertainty/depth/memory/novelty/history/exploration ablations are exposed; no depth superiority is asserted.
- Browser QA passed: candidate table, next/previous/slider/play, JSON download, responsive layout and no page errors. Browser controls are checked separately by `tests/browser.cjs` (optional Playwright/Chromium). The managed browser blocks `file://`; a localhost static server is used for that QA, with no external assets required.

## Remaining scientific and practical work

1. Choose a compatible semantic detector and monocular depth checkpoint, pin revisions, test real CPU inference and quantify detection limitations.
2. Audit actual COCO-Search18 / free-viewing datasets, licenses, display transforms and official image splits. No human data has been run here.
3. Fit/calibrate on training images only, reserve untouched test scenes, check blur/grid/foveal-size sensitivity, and report participant/image uncertainty rather than seed-based significance.
4. Characterize association and uncertainty on cluttered photographs. Covariance/quality are heuristic; occlusion reasoning, perturbation consistency and metric RGB-D are not implemented.

M5 physical camera pose, robotics, head–eye physiology and Unity remain intentionally outside V1.

## Windows portability correction

Explicit UTF-8 added at all project text/CSV boundaries. A regression emulates cp1252, demonstrates the original template decode failure without explicit encoding, and verifies successful UTF-8 rendering. Pandas is imported only for evaluation, so optional Bottleneck warnings do not affect `run-one`. Cumulative suite: 30 passing tests; `run-one` and synthetic evaluation rerun successfully. Native Windows execution remains to be confirmed by the user.

## Phase 2 review implementation

The subsequent review is implemented in place. See `implementation-audit.md`, `phase2-changelog.md` and `reports/phase2/phase2-report.html`. Verified pretrained CPU inference on Porto and a COCO training photograph; 45 tests in both core and vision environments; actual-state browser geometry/selection checks; 60 human test trials, 6 internal test images, 10 participants, 3 seeds, 14 variants (2,520 runs). Real single/evaluation provider parity, future-fixation non-leakage and explicit no-depth/no-XYZ checks passed. Native Windows pretrained inference and population human likeness remain unproven.
