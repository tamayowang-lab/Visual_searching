# Autonomous real-environment observer

The current demonstration uses RGB photographs of Porto, the same panorama
source used in Gaze Lab. Three perspective views are checked in under
`examples/photos/`, with original-source hash and projection metadata. Earlier
shape-demo assets, their generator, color-name object categories and
color-specific attention/verification branches have been removed.

## Default execution and perception

`simulation.runner.run → autonomous.runner.run_autonomous` is authoritative.
Archived runners require `execution_path: archived`. Normal attention uses a
64-sample-wide dense field and seeded softmax (temperature 0.08), independent
of heuristic lognormal timing. Legacy fixation counts, viewing budgets, spatial
candidate grids and timing-race selection do not control this execution path.
The shared World update and saccade decay functions maintain object memory.

The existing pinned `ssdlite320_mobilenet_v3_large` detector supplies central
semantic object detections. A central circle retains full RGB input; peripheral
input is downsampled to 32 pixels wide before enlargement. The same detector
can produce tentative category cues from bounded peripheral input. Those cues
can affect task priority, but cannot enter central memory or verify the target.
No full-resolution semantic bank or supplied target boxes are used.

Memory updates at fixation offset. Before that commit, playback exposes only
pre-commit memory and the explicitly labeled last committed priority/assessment.
Exact captured PNG sensory frames are checked against the logged RGB SHA256.
During saccades, the retained sensory frame is labeled as such.

## Defaults and reproduction

See `configs/default.yaml` for the exact normal configuration: exploration,
until finished, seed 42, temperature 0.08, pinned SSDLite threshold 0.3 and
context scale 3, depth off, a 60000 ms administrative timeout, and absence
judgments disabled. Central radius defaults to 0.075 × image width. The real
photo demo uses radius 0.12 × width and a 20000 ms administrative timeout.

```bash
python scripts/setup_perception.py --download --objects-only
python -m active_observer.autonomous.demo --out outputs/autonomous --seeds 42 43 --serve
```

Open `http://localhost:8000/`. The generated folder contains `index.html`,
`replay.json.gz`, exact sensory frames, and `summary.json`. Serve the folder
through HTTP; all relative assets must accompany it. Static hosted playback
uses the same records. Arbitrary-seed regeneration requires the local server.
To use your own photograph:

```bash
active-observer run-one photograph.jpg --mode search --target person --out outputs/search
```

## Recorded real-photo results

Seed 42, target `person`:

| Real environment view | Exploration | Search |
| --- | --- | --- |
| Porto square | 59 fixations, administrative timeout at 20 s | 30 fixations, verified at 12.225 s |
| Porto promenade | 59 fixations, administrative timeout at 20 s | 5 fixations, verified at 2.196 s |
| Porto waterfront | 59 fixations, administrative timeout at 20 s | 5 fixations, verified at 2.211 s |

Seed 43 also verifies the person target in all three views (33, 23 and 33
fixations respectively). These are model verification outcomes, not externally
validated detection correctness. Exploration timeouts are explicitly
`cognitive_completion: false`. They are not successful cognitive completion.
All six exploration runs extend beyond 16 fixations.

## Verification and limits

Focused checks cover default dispatch, lack of count/grid dependence, valid
coordinates, contribution sums, exact sensory frames, fine-detail gating,
causal memory, fixed-time prefixes, reproducible histories, and central-semantic
verification. Browser checks cover the three photographs, playback and
inspection, person search, seed changes, local regeneration, and mobile layout.
The current executable test is `tests/browser_autonomous.cjs`; unit tests are
in `tests/test_autonomous.py` and the relevant archived regression files.

Stopping thresholds, expected gain and confidence remain heuristic and
uncalibrated. SSDLite sometimes misidentifies objects, especially from blurred
input. These uncertain predictions are displayed as actual model state.
No identities are corrected by hand for the replay. Target absence is disabled
by default. Peripheral hypotheses never establish target verification. Depth
is not used. No full Phase 4B/5 validation or human evaluation pipeline was run.

The exact changed paths are in `docs/autonomous-changed-files.txt`.

Current verification: 46 focused unit/regression checks passed; all 11 browser
checks passed, including actual Python regeneration with seed 777. All 12
recorded trajectories were audited for central-semantic-only memory, frame
counts, and 36 sampled exact sensory-frame hashes.
