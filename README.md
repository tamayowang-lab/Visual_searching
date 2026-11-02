# RGB Active Visual Observer

RGB means an ordinary RGB photograph of a real environment. The observer
searches the scene through gaze-dependent visual input, remembers centrally
acquired object hypotheses, and records why it looks somewhere or stops.

The default browser demonstration uses the **Porto riverside photograph from
Gaze Lab**, projected into three perspective views. All gaze trajectories,
sensory frames, priority maps, object memory, and decisions come from actual
Python simulation runs. No object boxes or scene labels are provided to inference.

## Open the demonstration

Published private viewer:
<https://active-observer-visual-demo.tamayowang.chatgpt.site>

It supports exploration and goal-directed search for a person, recorded seed
selection, playback, fixed-time inspection, priority-component inspection,
central object memory, tentative peripheral cues, and stopping explanations.
The hosted version replays recorded Python results; it does not execute Python.

## Local setup and launch

Python 3.11 or newer:

```bash
python -m pip install -e '.[test]'
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python scripts/setup_perception.py --download --objects-only
python -m active_observer.autonomous.demo --seeds 42 43 --serve
```

Open <http://localhost:8000/> on the machine running that server. The local
viewer can regenerate arbitrary seeds through its Python endpoint. The three
prepared photographs are in `examples/photos/`. To recover their original
panorama and reproduce the perspective projection:

```bash
python scripts/prepare_photo_demo.py --download
```

For your own real environment photograph:

```bash
active-observer run-one photograph.jpg --out outputs/scene
active-observer run-one photograph.jpg --mode search --target person --out outputs/search
active-observer run-one photograph.jpg --condition fixed_time --duration 15 --out outputs/15-seconds
```

Open the generated `inspector.html`. The full autonomous history is saved as
`run.json`; fixed-time runs additionally save a causal `fixed-time-view.json`.

## Current model

Normal runs use `simulation.runner.run → autonomous.runner.run_autonomous`.
`configs/default.yaml` selects the existing CPU SSDLite detector, heuristic
lognormal timing, and dense spatial softmax selection (temperature 0.08).
The autonomous loop has no fixation-count limit and ignores the legacy attention
candidate grid and legacy viewing budget. A 60-second administrative timeout
protects normal runs; demonstration recordings use 20 seconds.

Central input retains the observed RGB pixels. Peripheral input is reduced to
32 pixels wide before enlargement. The existing SSDLite model processes this
bounded peripheral input for **tentative** category cues, which can guide
search but cannot verify a target or enter central semantic memory. Only
centrally gated detections update object identity memory and count toward
repeated-evidence target verification.

Dense priority shows salience, peripheral support, novelty, uncertainty,
expected information gain, semantic task relevance, movement cost, and
inhibition of return. Each choice and stopping assessment is recorded.
Fixed-time and until-finished inspection use the same autonomous history.
Confidence, information and expected-value scores are heuristic, uncalibrated
model estimates. A detector prediction can be wrong; the viewer displays its
actual hypotheses rather than supplying corrected identities.

Historical phase runners remain available through `execution_path: archived`.
Their documents and statistical reports are historical experiments, not the
current default. The lightweight structural provider proposes unlabeled
contrast structures and never claims semantic object recognition.

## Checks and scope

See `docs/autonomous-repair.md` for architecture, current demonstration results,
and focused verification. No full Phase 4B/5 evaluation pipeline or new dataset
was run for this repair. The existing SSDLite model was reused; no new network
architecture was added. Depth remains outside the current demonstration.

The Porto source is the Salient360 toolbox's `1_PortoRiverside.png`:
<https://github.com/David-Ef/salient360Toolbox>. Source hashes, projection angles
and provenance are recorded in `examples/photos/manifest.json`. The repository
has an MIT license; no separate stimulus-specific license was documented.
