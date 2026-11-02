# Active visual observer for gaze-dependent 3D scene reconstruction

This document is the authoritative scientific scope of this branch. The target
system is an interpretable, biologically inspired observer that sequentially
samples a single real RGB environment photograph and incrementally constructs
an uncertain hierarchical 3D scene representation.

The input is not a complete, equally detailed scene representation. Each
fixation supplies high-resolution foveal evidence and lower-resolution
peripheral evidence. Peripheral processing provides visual salience, tentative
structures, grouping, spatial continuity, and ensemble statistics. Foveal
processing provides detailed recognition, boundaries, and monocular depth.
**Depth inference must run on the current gaze-dependent input, not on the
complete original photograph followed by a visibility mask.**

Memory has distinct individual-object and ensemble levels. Individual objects
are represented by uncertain cuboid hypotheses whose position and dimensions
can be updated with observations. Ensemble knowledge describes a region or
group without implying that its individual members have been reconstructed.
Unknown scale, hidden surfaces, and ambiguous geometry must remain uncertain.

Attention combines bottom-up salience, grouping, ensemble evidence, novelty,
inhibition of return, and movement cost with top-down reconstruction needs.
The desired actions are exploring unfamiliar regions, inspecting individual
objects, and revisiting useful locations. A fixed number of eye movements is
not a stopping rule. The observer stops autonomously when the estimated benefit
of another observation becomes sufficiently small; a computational safety
interruption must be recorded separately.

The central hypothesis is that integrating peripheral ensemble perception,
focused object inspection, and uncertainty-guided attention improves the
efficiency and coherence of 3D scene representation relative to salience alone.
This hypothesis is not established by this branch or by an attractive replay.

## Three scientific distinctions

1. **Perceptual uncertainty versus reconstruction uncertainty.** Identity and
   2D localization uncertainty are separate from depth, size, and geometric
   uncertainty. Clear recognition does not establish accurate physical geometry.
2. **Ensemble knowledge versus individual-object knowledge.** A belief that a
   region contains approximately ten similarly sized objects is not ten
   reconstructed cuboids. Group counts, size distributions, and member identity
   evidence must have separate provenance and uncertainty.
3. **Estimated information gain versus actual reconstruction improvement.**
   Policy predictions are predictions. Without suitable reference 3D data,
   consistency and stability are measurable proxies, not geometric correctness.
   Unknown actual improvement must remain unknown.

## Implementation boundary

This branch is a cleaned foundation, not an implementation of the entire model.
See `docs/implementation-status.md` for the exact supported and missing pieces.
Categorical target-search experiments, target-present/absent decisions, human
scanpath benchmark machinery, timing-model comparisons, and historical phase
reports are outside this branch's current scope. Their previous implementation
is preserved on `archive/pre-reconstruction-cleanup`.
