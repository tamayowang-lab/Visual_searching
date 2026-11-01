# Phase 5 experimental-control audit

Phase 4B is preserved at public commit
`8a1af60db3e33ff4daa3088efefcde603b52d0f2`.

- `Config.max_fixations` is a normal stopping boundary in the archived runner.
- `time_budget_ms` is passed into fixation timing: different budgets can change
  terminal dwell, committed evidence, memory, candidate probabilities and RNG use.
- `stop_on_target` accepts a single central memory threshold, with no separate
  possible-target versus verification decision and no absence inference.
- There is no exploration completion rule or explicit abandoned-search outcome.
- The old evaluator conditions on each human response time and supplies an
  initial gaze; it cannot validate autonomous completion time.
- Earlier source fixation records lack explicit event onsets; cumulative duration
  must not be presented as measured elapsed time including saccades.
- COCO search is not free-viewing ground truth. Phase 4B pilot has only six new
  test scenes and cannot select a Phase 5 architecture.

Phase 5 will use a separate opt-in experimental runner. It will generate one
trajectory with a declared engineering timeout, record all motor transition
intervals, and expose fixed-time views without generating new random draws.
Partially observed fixations will not expose end-of-fixation evidence updates.
Administrative timeout will never be labeled successful cognitive completion.

The first human experiment is the same-image, 5-second COCO-FreeView versus
COCO-Search18 contrast. Search trials that end earlier are censored at their
actual endpoint; they must not be padded with invented gaze. If event timing is
insufficient for exact 5-second alignment, report the timing assumption and its
sensitivity, or restrict the primary endpoint to supported records.
