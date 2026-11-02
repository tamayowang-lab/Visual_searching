# Gaze-dependent reconstruction observer

The scientific definition is in [MODEL.md](MODEL.md). This branch is a clean
foundation for that model, with one execution path and explicit missing
capabilities. It is not a complete 3D reconstruction system or a validated model
of human vision.

## Scientific boundaries

- Recognition and 2D localization uncertainty are stored separately from depth
  and dimension uncertainty.
- Scene-level ensemble summaries are separate from individual object memory.
  Coarse connected structures are not an object-count estimate.
- Predicted information gain is separate from measured consistency proxies.
  Actual 3D improvement is `null` without reference evidence.
- Every depth inference receives current gaze-dependent RGB. The runner never
  supplies the complete original photograph to a depth estimator.
- Hidden thickness and unidentified scale remain unknown; no complete metric
  cuboid or aligned 3D fusion is claimed.

## Run

```bash
python -m pip install -e '.[test]'
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python scripts/setup_perception.py --download --objects-only
active-observer examples/photos/porto-square.jpg --config configs/default.yaml --out outputs/foundation
```

`configs/default.yaml` is explicitly a perception-only diagnostic until a depth
provider is configured. Its result reports unavailable depth and never claims a
completed 3D reconstruction. It is useful for checking the sensory, memory and
attention wiring without downloading depth assets.

For the retained MiDaS adapter:

```bash
python -m pip install timm==0.6.13
python scripts/setup_perception.py --download
active-observer examples/photos/porto-square.jpg --config configs/with-depth.yaml --out outputs/with-depth
```

This enables per-fixation depth inference, not cross-fixation scale alignment
or complete physical object recovery. Missing checkpoints fail explicitly.
No pretrained downloads happen inside the runner. The depth-enabled path was
checked with instrumented test sensors; this cleanup did not download or
validate the MiDaS weights.

Each run writes `run.json` and an exact-input diagnostic `inspector.html`.
Open the inspector locally. `--inspect-seconds 15` writes a causal prefix to
`inspection.json` without changing the autonomous trajectory. The only execution
limit is an administrative elapsed-time safeguard, normally 60 seconds.

## Small code surface

```text
perception/     gaze-dependent RGB, central object providers, depth adapters
world/          object association, separate reconstruction state, ensemble summaries
attention/      dense priority field and one lognormal timing heuristic
simulation/     sole runner, inspectable stopping, causal replay snapshots
viz/            diagnostic replay of actual states
```

Configuration has no phase dispatcher, fixation-count limit, target instruction,
predefined attention candidates, or timing-race variants. Unsupported old
configuration keys fail instead of silently changing behavior.

```bash
python -m pytest -q
```

The focused tests require no model downloads or human datasets. See
[implementation status](docs/implementation-status.md) and
[cleanup record](docs/cleanup.md). The earlier implementation and its reports
are recoverable from `archive/pre-reconstruction-cleanup`.

## Real photographs

`examples/photos/` contains perspective views of the Porto riverside panorama
used in the previous visual-searching demo. These are ordinary RGB environment
photographs. Their manifest records source hash and projection parameters;
it supplies no target labels or object boxes. Regenerate them with
`python scripts/prepare_photo_demo.py --download`.

Source: [Salient360 Toolbox](https://github.com/David-Ef/salient360Toolbox).
The repository has an MIT license; a separate stimulus-specific license was
not documented. No human evaluation data was downloaded for this branch.
