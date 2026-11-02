# Historical foundation cleanup

This record describes commit 472e40e. Subsequent repairs and the recorded 3D
laboratory are documented in laboratory.md and repair-audit.md.

# Reconstruction refocus

Working branch: `refactor/gaze-dependent-3d-observer`.
Preserved predecessor: `archive/pre-reconstruction-cleanup` (commit `950ba93`).
The predecessor captures all earlier tracked and untracked source changes from
the real-photo search demo, so removal here does not erase that work.

## Removed from the working tree

- Phase 2/3/4B/5 dispatchers and duplicate simulation runners.
- Categorical target search, target verification/absence controllers and mode selectors.
- Fixed candidate grids, alternate timing races, ablation configurations.
- Human scanpath import, benchmarking, statistical/report generation machinery.
- Historical reports, phase audits, old preview sites and their viewers.
- Tests that asserted the removed experiments' behavior.
- Unused robot-pose boundary utilities and plotting/table runtime dependencies.

## Retained and made explicit

- Real environment photographs and their provenance.
- Gaze-dependent input, central recognition adapters and existing depth adapters.
- A single runner that performs depth inference on actual fixation input.
- Object association, relative geometry and separate perceptual/reconstruction uncertainty.
- Distinct aggregate ensemble evidence; count/size estimates remain unavailable.
- A dense interpretable priority field, one timing heuristic and a small replay viewer.
- Separate predicted gain, measurable 2D consistency proxies and unknown true 3D improvement.

This is intentionally a foundation branch. It does not silently present the
old search observer as the complete reconstruction model. `MODEL.md` is now the
scientific source of truth and `implementation-status.md` records the gaps.

The previously published Site is a separate deployment of the earlier search
demonstration. Creating this branch does not replace it with an unfinished
reconstruction model.

## Checks performed

- 17 focused scientific-boundary and execution tests passed.
- All 23 remaining package modules imported successfully.
- A short real Porto photograph run used the existing central SSDLite provider
  and produced JSON, a causal 100 ms inspection, and an exact-input replay.
- Browser smoke checks passed for initial-state causality, fixation stepping,
  separated prediction/proxy and ensemble displays, administrative termination,
  mobile layout, and absence of JavaScript errors.
- `git diff --check` passed. No full statistical evaluation was run.

The optional depth input contract was verified with instrumented test providers.
A real pretrained MiDaS run and complete 3D reconstruction were not validated.
