# Phase 3 timing-only milestone

The user's narrower milestone takes precedence over the broader attached Phase 3 plan: compare lognormal, accumulator, and accumulator-plus-hold before changing peripheral guidance, spatial priors, EIG or other attention mechanisms.

## Preserved baseline and scope

`timing_model: phase2` still dispatches to the unchanged Phase-2 numerical loop (`simulation/runner.py:run_phase2`). Existing Phase-2 report artifacts are not overwritten. Spatial utility weights, `attention/policy.py`, perception providers, association, memory update and duration-dependent world-evidence expressions are unchanged. The new loop uses the same functions and commits memory once per terminated observation, avoiding tick-frequency-dependent familiarity or visits.

Three new matched variants share motor assumptions and the original softmax destination selector:

1. `heuristic_lognormal`: the existing context-dependent mean and 100–800 ms bounds, with train-fitted scale and lognormal spread. Initial dwell retains the scaled fixed-250 ms convention.
2. `accumulator`: a noisy competing first-passage readiness model, without additional processing hold.
3. `accumulator_hold`: the identical readiness model, with a separately switchable bounded hold. Hold strength zero reproduces the no-hold trajectory exactly under the same noise stream.

This comparison estimates the **incremental effect of the new timing hold**. The existing bounded stay option remains shared across all three spatial policies; it is not silently removed. Consequently, “no hold” means no *additional readiness inhibition*, not a claim that the whole observer has no processing dependence.

## Equations and units

Fixation processing is integrated with an initial 20 ms step, converted to seconds for diffusion. For onset-snapshot utility values U_i, define relative priority r_i=(U_i-min U)/(max U-min U), or zero for a uniform vector. The existing utility already includes movement cost; it is not charged twice.

```
A_i(t+dt) = max(0,
  A_i(t) + [b + g*r_i + urgency_ramp*t] * [1-H(t)] * dt
           + sigma * sqrt(dt) * Normal(0,1))
```

The readiness threshold is fixed at 1 during fitting to avoid fitting an unidentifiable drift/threshold scale. The first threshold crossing triggers a separate motor-preparation stage. Within-step linear crossing interpolation reduces discretization error; it is not an exact Brownian-bridge crossing solution. Nonnegative reflection is a modeling assumption.

The online processing proxy is:

```
e(t) = ceiling - (ceiling-e0)*exp(-t/tau)
gain(t) = (ceiling-e0)*exp(-t/tau)/tau
H(t) = hold_strength * clip(gain(t)*normalization_tau, 0, 1)
```

Ceiling and tau depend on the *current* central observation's visibility, region size, local image quality and recognition ambiguity; detector confidence is one input, not the latent evidence state. A low-confidence nonsemantic texture proxy is used when no central object is detected. The proxy saturates and is evaluated online. It is the actual internal signal inhibiting readiness, not a post-hoc illustrative curve.

**Important boundary:** this milestone does not replace Phase-2 world-confidence dynamics. The timing proxy is a provisional, separate estimate of current processing. World beliefs are committed with their original function once the fixation ends. Full short-step belief assimilation, new peripheral category evidence, new EIG and center-prior changes are deferred rather than bundled into this timing test.

## Where versus when

The primary comparison uses `selection_mode: softmax`. The race controls **departure readiness**; the existing spatial policy is rescored after processing and its softmax selects the actual destination. The first-ready candidate and executed destination are both logged and need not coincide. This isolates timing from a new spatial winner-selection rule.

An optional `selection_mode: race` executes the first-ready plan. It is independently selectable and is **not** part of the primary three-way experiment. Priorities are onset snapshots; accumulator states and processing hold evolve during the fixation. No new full-resolution peripheral semantic access is introduced.

## Event accounting

- Decision time: threshold crossing (or sampled release for the lognormal baseline).
- Motor preparation: a separate positive latency after release, while gaze remains stationary and processing can continue.
- Fixation duration: decision plus completed preparation, subject to external-window right censoring.
- Saccade execution: separate from fixation duration. The provisional pixel formula is 20 + 0.04 × amplitude_px ms, bounded to 10–90 ms; a declared PPD permits a separate angular heuristic. These are explicit assumptions, not estimates from fixation-only recordings.
- No sensory observation or grid coverage is acquired during execution; memory only decays.
- Next fixation onset follows execution. A window can end during decision, preparation or execution; each case is logged. A 3-second numerical safety timeout is labeled, not silently treated as a physiological maximum.

Races are not clipped to the lognormal baseline's 100–800 ms range. Same-location observation segments are merged by the existing evaluation helper, and censored terminal observations are distinguished from complete fixations.

## Fitting and evaluation protocol

Population-level fitting is used because there are only 12 training scenes and sparse within-person complete fixations. Threshold, priority gain and sensory time constant are fixed; small grids fit urgency, diffusion, preparation, hold strength, and baseline scale/spread. Conditional CRPS is averaged within image/participant and then across images. Common random numbers reduce Monte Carlo comparison noise; six predictive samples are an approximation.

Conditional training/validation contexts use observed *past* human gaze and the current observation, never the next fixation. Initial and terminal human fixations are excluded from timing fitting; terminal intervals may be response-censored. Test conditional timing scores are labeled separately from autonomous scanpath predictions.

For each variant, training chooses its parameters. Validation chooses the simplest model within one standard error of the best validation image-mean CRPS. The held-out test is inspected only after the selection file is frozen. These are the **previously examined Phase-2 internal test images**, not a new untouched confirmatory sample.

The autonomous comparison uses the same six internal test scenes, 10 participants, 60 trials and common seeds. Human RT is supplied as the exposure window, not predicted. Report duration distributions, amplitude distributions, NSS, custom spatial DTW, revisits, fixation counts and censoring. Paired crossed image/participant bootstrap intervals average model seeds first. They are approximate pointwise intervals with only six scenes, not multiplicity-adjusted proof of superiority.

Fixation-only data cannot identify decision, preparation and execution components independently. Good total-duration fit would not validate their biological interpretation or establish that the brain uses a competing race. Negative and tied results are retained; the recommended architecture is the smallest supported by validation and joint temporal/spatial evidence.


## Observed outcome

The completed exploratory test contains 60 trials × 3 seeds × 3 variants. Conditional test timing CRPS (ms, lower better) is 56.89 lognormal, 62.75 accumulator and 61.23 accumulator+hold. The validation simplicity rule selected accumulator, but its ranking did not generalize to conditional test timing.

Autonomous duration Wasserstein errors are 87.18, 73.79 and 76.19 ms. Accumulator vs lognormal change is −13.38 ms with approximate crossed-bootstrap 95% interval [−28.03, +0.62]; hold change is −10.99 ms [−24.96, +1.79]. Neither interval excludes no improvement. Spatial NSS remains 0.092–0.100, custom DTW about 0.180–0.183, and amplitude mismatch remains large. Additional hold is not supported by these results. Keep the archived default, retain all timing options, and seek a new, larger scene sample before promotion. See [aggregate report](../reports/phase3/phase3-report.html) for distributions, censoring and all paired intervals.
