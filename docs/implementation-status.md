# Status against the proposed model

| Component | Present foundation | Missing or limited |
| --- | --- | --- |
| Sensory acquisition | Central RGB plus bandwidth-limited peripheral input, exact input hashes | Calibrated visual angle and a graded biological eccentricity profile |
| Peripheral grouping/ensembles | Global/local coarse moments, edge density, structural-group counts and sensory reliability; separate histories | Object count and size distributions; grouping is a proxy, not object enumeration |
| Foveal objects | Existing centrally gated detector adapters, association, evidence history | Recognition calibration, reliable boundary completion |
| Gaze-dependent depth | Configured provider invoked on each current sensory image; input hash logged; central evidence integrated | Real pinned MiDaS + SSDLite diagnostic recording generated; no accuracy validation |
| Object geometry | Explicit camera-Z proxy, half-open bbox projection, full separate observation histories and open front rectangles | Hidden thickness, orientation, physical scale, complete extent |
| Reconstruction uncertainty | Separate depth and per-dimension uncertainty; recognition cannot reduce these | Calibrated geometry uncertainty and principled fusion |
| Hierarchical memory | Distinct object registry and ensemble-summary history | Learned ensemble-to-member relationships and robust long-term fusion |
| Attention | Dense salience, novelty, peripheral structure, perceptual need, geometry need, IOR, movement; seeded softmax | Proper expected reconstruction-improvement estimator, ensemble size/count-driven policy, action-level explore/inspect/revisit model |
| Timing | One independent heuristic lognormal dwell | Human timing calibration |
| Stopping | Inspectable persistent low-estimated-value criterion; administrative timeout distinct | Validated benefit threshold and evidence that internal stopping reflects sufficient reconstruction |
| Evaluation | Observable 2D association consistency; explicit `actual_reconstruction_improvement: null` | Aligned depth-stability measures, annotated 3D evaluation, salience-only hypothesis comparison |

## Interpretation rules

A successful detector output is not a successful reconstruction. An ensemble
summary does not create individual objects. A lower predicted next-fixation
value is not measured geometric improvement. Consistency can also occur for
consistently wrong beliefs.

The current gain formula is explicitly a heuristic scheduling proxy. Remaining
geometric uncertainty does not imply it can be reduced from the same single
view. Hidden thickness stays maximally uncertain and does not itself justify
endless revisits. A bounded exposure heuristic discounts predicted reducibility;
it is not a fitted uncertainty-reduction law.

The depth adapters normalize each gaze-conditioned prediction independently.
The new runner deliberately does not average those depths as though they shared
a verified metric scale. Each update records its scale identity when available;
without alignment, depth stability and actual 3D improvement remain unknown.

The default configuration does not substitute constant or precomputed full-image
depth when a model is unavailable. It reports the missing capability. The
explicit depth configuration raises an error for missing assets.

`low_estimated_benefit` is an internal stopping outcome, not validated scene
understanding. Without a depth provider, low proxy value yields
`reconstruction_unresolved`. `administrative_timeout` is never reported as
completion. Every result carries `reconstruction_validated: false`.

## Next implementation order

1. Establish gaze-conditioned depth scale/reference handling and boundary evidence.
2. Implement ensemble count/size uncertainty without fabricating individual members.
3. Specify full cuboid priors and fusion, with independent geometric uncertainty.
4. Define observable reconstruction-consistency/stability proxies and reference-data boundaries.
5. Connect predicted reconstruction improvement to action selection and stopping.
6. Only then compare the stated hypothesis with a salience-only baseline.

No full statistical validation pipeline, human dataset download, timing-model
comparison, or claim of hypothesis confirmation is part of this cleanup.

## Repair-first laboratory update

Phase A passed its 35-test gate before Three.js implementation. The final suite
has 37 focused tests. Camera depth now means Z, not ray range; per-fixation NPZ
assets retain exact arrays and provenance; all geometry observations retain
their independent frame IDs, corners and uncertainty. No alignment is claimed.
Derived gain is no longer added again on top of its constituent priority cues.

The lab supports linked 2D/3D selection, independent frame groups, camera frusta,
orbit/zoom, stored depth/quality/validity maps, localized ensemble inspection,
all priority layers, fixed or labeled per-frame scales, causal time/commit
controls, before/after memory comparison, and separate prediction/proxy plots.
Actual 3D improvement stays null. Disabled depth and empty memory stay explicit.

Browser QA compared 39 Python/JavaScript snapshots across seven cases and checked
actual 3D picking/rendered corners, map values, timing, comparisons, downloads,
mobile layout and zero external requests or JS errors. The real diagnostic used
the existing pinned providers on Porto RGB. Mocks are not model evidence.
See laboratory.md and repair-audit.md for exact limits and reproduction.
