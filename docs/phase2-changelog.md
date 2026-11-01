# Prioritized phase-2 change log

## P0: make perception real and prevent leakage

- Verified pinned SSDLite and MiDaS Small CPU providers, explicit checksum verification and offline-first setup; no silent semantic/depth fallback.
- Unified single-image, preview and evaluation provider selection and cache reuse.
- Kept no-depth/classical baselines; preserved centrally gated observations and no future-human-fixation input.
- Measured first/cached inference and whole-process peak memory on actual photographs.

## P1: coherent memory and finite evidence

- Separate identity, image-position and depth uncertainty, depth dispersion, observation count, source/partial flags, image covariance and relative 3D covariance.
- Image-anchored position and observed-surface bounding envelopes; separate future rigid-transform interface.
- Stronger association gates and partial-extent preservation.
- Duration-dependent bounded support with a preserved legacy evidence update.

## P2: interpretable policy and timing

- Expected sensory reliability / expected uncertainty reduction / observation-cost terms, with old uncertainty×novelty as a named baseline.
- Independent novelty term, switchable information ablation, bounded stay decision and merged stay segments for evaluation.
- Existing stochastic softmax and bounded lognormal duration retained; no complex drift-diffusion system added.

## P3: actual-state visualization and validation

- Rotatable actual-memory geometry, linked object selection, observed-only depth and separately labeled private estimator reference.
- Real COCO-Search18 target-present data with audited display transformation, independent internal image splits, recorded-duration conditioning, multiple seeds, baselines and ablations.
- Controlled mechanism tests, empirical/perception reports, explicit limitations and no population-human-likeness claims.

Implementation stages were checked incrementally: original 30 tests; perception factory compatibility 30; memory/geometry 34; policy/stay 38; controlled perception/cache tests 43. Final checks and empirical counts are recorded in the phase-2 report. Empirical results are descriptive and do not justify selecting parameters on these test images.
