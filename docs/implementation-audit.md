# Implementation audit — phase 2

Audit basis: existing RGB Active Observer V1 on `feature/rgb-active-observer-v1`, extended in place. No unrelated parent-repository directories are changed. Prior working CLI, offline viewer, prepared-run site, seeds and Windows UTF-8 handling are retained.

## What V1 actually did

| Mechanism | Original entrypoint | What was used in ordinary inference | Review finding |
|---|---|---|---|
| Object proposals | `perception/objects.py:ClassicalObjects.detect` | Connected color components on foveated RGB | Color names, not semantic recognition |
| Optional semantics | `FovealTransformersObjects.detect` | Only when explicitly selected | Had not been verified with weights |
| Depth | `perception/depth.py:ConstantDepth/ArrayDepth/PrecomputedDepth/TransformersDepth` | No depth by default; optional HF adapter unverified | Private map contract existed; no real pretrained demonstration |
| Foveation | `perception/foveation.py:observe` | Gaze-dependent RGB blur and reliability | Pixel approximation unless PPD supplied |
| Memory | `world/model.py:World.update` | Object registry, coarse grid, confidence/novelty/IOR | Confidence updated once per observation, not duration-dependent |
| Association | `world/association.py:associate` | Hungarian matching using spatial/category/appearance costs | Weak gates allowed identity merging; boxes overwritten |
| Geometry | `perception/depth.py:update_observed_depth` | Central surface samples only | XYZ proxy existed but covariance mixed pixel and relative units |
| Attention | `attention/policy.py:score/choose` | Explicit utility and seeded softmax | Information was uncertainty × novelty; novelty counted separately too |
| Timing | `attention/dynamics.py:duration_ms` | Bounded lognormal dwell | Uncalibrated; no explicit marginal observation value |
| Run loop | `simulation/runner.py:run` | Sequential static-image gaze updates | No eye physiology or novel viewpoints; these remain non-goals |
| Metrics | `evaluation/metrics.py:compare` | NSS, tie-correct AUC, custom spatial DTW and summaries | Custom measures correctly labeled, but no actual human run |
| Evaluation | `evaluation/batch.py:predict/evaluate` | Classical objects, optional record depth | Could not use the same configurable semantic/depth factory as run-one |
| Visualization | `viz/dashboard.py`, `viz/html.py`, `preview/` | PNG/offline HTML, prepared Python-run browser replay | No rotatable actual-memory geometry |

## Current inference path and exact ownership

1. `config.py:Config` selects explicit model/evidence/information options; `perception/factory.py:ProviderFactory.get` is shared by `runner.run`, the single-image CLI, preview builder and `evaluation.batch.predict`.
2. `perception/foveation.py:observe` produces actual blurred RGB. Semantic input is a **foveated context crop**, not the original full-resolution photograph.
3. `perception/pretrained.py:TorchvisionFoveal.detect` runs SSDLite320 MobileNetV3 on CPU. Only detections whose centers are central leave the adapter; their boxes are clipped to the local central region. Classical peripheral cues remain explicitly nonsemantic.
4. `perception/pretrained.py:CachedObjects.detect` caches complete foveated observations, gaze, field radius and provider fingerprint. It does not build a hidden full-image semantic proposal bank.
5. `perception/pretrained.py:MidasSmall.predict` produces a private normalized inverse-depth map. Checkpoint SHA256, MiDaS source revision and source-file digests are recorded. `CachedDepth.predict` avoids repeated image inference. `update_observed_depth` restricts memory updates to central valid surface samples.
6. `world/model.py:World.update` retains identity and observed extent, observation counts, last-seen time, familiarity, soft IOR, visibility and separate identity/position/depth uncertainty. `evidence_model=bounded` uses `1-exp(-duration/evidence_time)` support, capped by measurement reliability. `legacy` preserves the prior update for comparison. Repeated identical evidence cannot make confidence exceed its sensory ceiling.
7. `world/association.py:associate` uses overlap, distance, appearance and class-support gates. Conflicting confident semantic classes cannot merge. Partial detections retain the union of **previously observed** extents. No unseen whole-object shape is inferred. This is not a robust multi-object tracking benchmark result.
8. `perception/depth.py:update_observed_depth` records dispersion, relative XYZ and a relative 3D covariance/envelope. Image covariance is stored separately in pixel units. Bounding envelopes cover observed image extents plus an uncertainty proxy, not complete physical volumes.
9. `world/frames.py:CameraPoseProvider/transform_point` is a tested future rigid-transform interface, **not used during current static-image inference**. All current positions are image-anchored, not robot/world coordinates.
10. `attention/policy.py:score` computes an efficient expected-reduction proxy from current uncertainty, anticipated sensory reliability, a finite observation horizon and an explicit observation cost. Novelty remains a separate term. `information_model=uncertainty_novelty` preserves V1. Both are heuristics, not calibrated mutual information. `choose` remains seeded softmax.
11. `score` adds a stay candidate only with a remaining sensory evidence gap and a bounded number of consecutive stays. `runner.run` logs stay/shift/stop decisions. `evaluation.metrics.merged_model_fixations` merges additional stay segments before fixation metrics.
12. `evaluation/batch.py:evaluate` shares provider instances/caches across runs, applies explicit recorded viewing duration where supplied, includes center/salience/random-walk baselines and semantic/depth/memory/uncertainty/novelty/history/exploration/information ablations. It reports image and participant summaries after seed averaging.
13. `preview/app.js:drawGeometry/drawDepth` renders **the actual memory snapshot** used by the controller. Selection links RGB boxes, rows and uncertainty envelopes. No separate reconstruction, floor mesh, unseen room or fabricated object model exists.

## Demonstrated versus untested

- Demonstrated: real CPU SSDLite/MiDaS inference on photographs, central semantic/depth gating, pinned checkpoint identities, cached repeated observations, bounded evidence, association regression tests, seed reproducibility, no-future-fixation leakage tests, rigid-transform validation, controlled stimuli, Windows-codepage regression, and browser geometry interactions.
- Scientific hypotheses: utility weights, sensory confidence, expected information gain, memory decay/familiarity/IOR, duration parameters, relative depth quality/covariance, and stay value. They are not identified physiological or human parameters.
- Not established: semantic accuracy, metric depth accuracy, complete shape reconstruction, population human likeness, robot coordinates, eye/head dynamics, robust occlusion tracking, native Windows execution of new pretrained models, or generalization beyond the limited empirical pilot.
- The optional Hugging Face adapters remain alternative unverified checkpoints. The **torchvision/MiDaS profile**, not arbitrary HF models, is the verified pretrained path.

## Known approximation boundaries

The default fovea is pixel-based until PPD is configured. MiDaS normalization depends on whole-image scale statistics but only local samples are acquired as beliefs; its private reference map is never a priority map. Quality is a fixed proxy plus local dispersion, not calibrated uncertainty or perturbation consistency. Expected information reduction is optimistic when identity ambiguity cannot be resolved by another look. Association can still duplicate an object after severe crop changes or incorrectly link similar adjacent objects. Partial extents can over-union after an association error. The fixed scene has no true occlusion dynamics.
