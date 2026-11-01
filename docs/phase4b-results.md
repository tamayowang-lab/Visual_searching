# Phase 4B results: what the mechanisms contribute

Six new test scenes, 58 correct target-present human trials, three simulation
seeds, eight variants (1,392 autonomous runs). These scenes were disjoint from
observer development scenes, but come from released COCO-Search18 training
material and may overlap vision-backbone training. This is an exploratory
study with limited scene-level power, not the official benchmark test.

| Variant | Target acquired after initial gaze | Count NSS | Mean saccade, px | Padding fixations |
|---|---:|---:|---:|---:|
| A original | 12.2% | 0.099 | 176.8 | 13.8% |
| B valid stimulus | 11.1% | 0.122 | 163.5 | 0% |
| C target guidance | 13.3% | 0.145 | 163.0 | 0% |
| D revised selection | 15.0% | 0.676 | 127.9 | 0% |
| E adaptive proposals | 14.0% | 0.275 | 100.8 | 0% |
| F persistent proto-objects + coarse sensor | 33.2% | 0.551 | 140.4 | 0% |
| S coarse sensor only | 15.4% | 0.251 | 112.1 | 0% |
| N proto-objects without persistence | 29.9% | 0.580 | 135.4 | 0% |

The corresponding human post-initial target-box acquisition rate is 79.2%,
not 100%, after the declared response-window reconstruction and exclusion of
the supplied initial gaze. Initial-on-target sensitivities are in the report.
No variant reproduces the full human behavior.

The most defensible conclusions are:

1. **Masking fixes technical invalidity.** It removes padding destinations
   without treating genuine dark image content as invalid. The 174 regenerated
   baseline runs have exactly the same fixation coordinates/durations as their
   earlier versions; only honest validity metadata was added.
2. **Selection policy addresses a real allocation problem.** At human prefixes,
   D versus C increases mass within 2° of the next human fixation from 2.0% to
   10.8%, and reduces expected next-fixation error from 204.5 to 163.8 px.
   The paired difference in expected error is −40.8 px, crossed-bootstrap 95%
   interval [−58.4, −25.7]. Autonomous count NSS also improves in its point
   estimate, but the D−C NSS interval is very wide [−0.343, 2.105].
3. **Coverage alone is insufficient.** E increases 2° candidate coverage from
   45.9% to 70.3%, but mass near human fixations falls from 10.8% to 5.5% and
   autonomous NSS decreases. Candidate refinement remains optional.
4. **Proto-object proposals warrant further study; persistence is unproven.**
   F reaches 33.2% target acquisition, compared with E's 14.0% and the sensor-
   only S control's 15.4%. However, F−S is +17.8 percentage points with interval
   [−3.3, 38.3]. The no-persistence N control reaches 29.9%; F−N is only +3.3
   points [−1.7, 12.5], while NSS changes by −0.029 [−0.175, 0.104]. These data
   do not establish a specific benefit of persistent tracking.
5. **Timing complexity is not independently vindicated.** All three original
   timing variants remain testable. For D, conditional CRPS is 60.93 ms for
   lognormal, 66.99 for accumulator and 59.68 for accumulator + hold. These are
   approximate six-draw scores with original frozen parameters; small ranking
   differences are not evidence for a biological mechanism.

Intervals are pointwise crossed scene/participant bootstrap intervals. Scene
sign-flip tests have only 64 possible sign patterns. No endpoint survives the
reported Holm correction across its full comparison family. Positive point
estimates and intervals should not be described as confirmed mechanism effects;
nonsignificance also does not prove equivalence. D remains the provisional
validation-selected comparator; no model was selected again on this test.

Keep D as a transparent simpler comparator and F/N as experimental alternatives.
The next useful scientific step is a larger, independently held-out scene sample,
with target-absent trials and separately collected free-viewing data, before
promoting persistence or a timing mechanism. Full object accuracy, semantic
calibration and human memory are not identified by these target-only records.

See the [quantitative report](../reports/phase4b/phase4b-report.html),
[paired statistics](../reports/phase4b/phase4b-paired.json),
[prefix effects](../reports/phase4b/phase4b-prefix-paired.json),
[method and trace contract](phase4b.md), and the
[owner-only decision laboratory](https://rgb-active-observer.candybike.chatgpt.site/phase4b/).
