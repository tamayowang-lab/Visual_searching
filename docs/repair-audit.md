# Repair-first laboratory audit (schema 2)

Phase A gate: all 35 focused tests passed before the Three.js viewer was added.

- Sensory input is hashed before provider calls and protected against mutation.
  Central semantic gating uses continuous gaze distance, not rounded pixels;
  observed boxes are clipped at foveal/image boundaries and marked partial.
- Camera coordinates are x-right/y-down/Z-forward. Normalized inverse depth
  maps to the declared nonmetric camera-Z proxy 1/(0.1+D), not ray range.
  Continuous half-open bbox edges may reach W or H.
- Every depth observation has a unique frame/local-scale ID. A provider's
  claimed scale ID does not establish alignment. All prior geometry, corners,
  evidence times, input/depth hashes and uncertainties are retained separately.
  Latest fields are compatibility aliases, not a fused world estimate.
- Exact depth, validity and quality arrays retain original dtypes and hashes.
  save_run writes compressed NPZ assets and provenance referenced by run.json.
  Re-export verifies hashes without inference.
- Connected coarse components expose localized boxes, RGB moments, edge density
  and mean reliability. These are structural groups, not object counts.
- Signed priority contributions include grouping separately. Derived gain is a
  diagnostic and stopping input, not added again to total priority. This repairs
  double-counting; it is not a policy change made to suit a visualization.
- Replay interpolates saccades with their planned duration, including censored
  saccades, and decays memory in gaps. Before fixation offset, new depth, memory,
  attention and stopping remain unavailable.

Limits: depth remains unaligned; hidden thickness, object orientation, metric
scale and actual 3D improvement remain unknown. No fusion or calibrated stopping
probability is introduced.
