# Phase 5 checkpoint: not a completed validation

This checkpoint preserves work before replacing the execution environment to
change its network policy. Phase 4B reports and the published Decision Lab remain
unchanged. Phase 5 is opt-in through `Config.experiment.enabled`; it is not the
software default and no new empirical result is claimed.

Implemented so far:

- Audit of legacy budgets and response-window conventions.
- Separate experimental runner with exactly explore/search modes.
- One autonomous source history, independent of requested fixed-time view.
- Explicit engineering timeout and optional administrative step safeguard.
- Heuristic expected-value controller, possible-target versus repeated central
  verification, coverage-based absence judgment, and unresolved outcome.
- Causal `inspect_at`: partial fixations expose pre-commit memory; snapshots
  include modeled transition time and preserve early terminal states.
- Regression tests for trajectory equivalence, no future-state leakage,
  transition-time accounting, terminal preservation and administrative limits.

Still required before release:

- Obtain and inspect official COCO-FreeView and target-absent search records.
- Strict dataset adapters and same-image 5-second protocol with real timing
  metadata; prespecify larger scene splits and primary outcomes before scoring.
- Run the matched human task contrast first. Do not use search gaze as freeview.
- Then controlled stopping experiments, calibration/false-positive/false-negative
  reports and held-out threshold checks. Current stopping parameters are unvalidated
  design assumptions, not selected or calibrated on any human data.
- Extend the website with the complete 2 × 2 controls and causal snapshots.
- Publish Phase 5 quantitative results only after actual execution.

Official data host `vision.cs.stonybrook.edu` currently returns a proxy tunnel
HTTP 403 in this task. Official FreeView page:
https://sites.google.com/view/cocosearch/coco-freeview
Search page (contains a target-absent download link despite outdated intro):
https://sites.google.com/view/cocosearch/

Network settings: Settings → Codex Cloud → Environments → … → Edit → Public
internet access → Additional allowed domains → `vision.cs.stonybrook.edu`.
Save, Republish, and use a new task attached to the updated environment. Existing
tasks can retain their old network policy. Workspace Agent Security can impose
additional restrictions. Do not edit the in-VM policy file to bypass enforcement.
