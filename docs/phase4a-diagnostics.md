# Phase 4A — diagnose before changing the observer

The [diagnostic report](../reports/phase4a/phase4a-report.html) answers the user's question at multiple spatial tolerances. The [private hosted report](https://rgb-active-observer.candybike.chatgpt.site/reports/phase4a-report.html) renders the plots directly. No production attention, perception, timing, world-memory, candidate or parameter implementation changed. Earlier reports and configurations remain archived.

## Main findings

On the six previously examined internal test scenes, existing candidates cover 95.7% of next human fixations within 48 pixels, but only 12.3% within 16 pixels and 33.3% within approximately 2°. The closest candidate's mean rank is 20.4, probability 3.46%, and distance 29.4 pixels. Probability within 48 pixels is just 5.31%. Thus broad representational coverage does not translate into correct selection; fine endpoint coverage is also limited.

At observed human prefixes, expected next-location error is 221.3 pixels, versus a nearest-candidate floor of 29.4. An evaluation-only future-coordinate candidate with the nearest existing utility reduces expected error only to 214.6 pixels. Greedy selection reduces it to 160.7, still far from the floor. These are conditional diagnostics, not autonomous forecasts or a new inference policy.

The original coordinate transformation is internally consistent. Including the first transition changes the autonomous observer/human amplitude comparison from 197.2/73.1 pixels to 198.5/115.7; it does not remove the discrepancy. Raw fixation-duration sums disagree with RT, and exact event onset alignment is unavailable. No locations were removed by the legacy RT clipping in this subset, so these timing caveats do not explain the amplitude discrepancy through dropped spatial endpoints.

Task guidance is incomplete: first-transition task utility is zero in the test subset, context contributions are zero with empty priors, and human target acquisition is 100% versus 18.3% of observer runs in this selected correct-response subset. Existing movement-prior and center baselines improve some movement/sequence metrics while worsening target acquisition. They were not promoted to replacements.

A concrete preprocessing/candidate artifact is also confirmed: 13.8% of prefix probability and 15.9% of autonomous observer fixations fall in artificial display padding, versus no human fixations. This is distinct from unknown scene background. It was measured without modifying the production candidate set.

## Reproduce without modifying inference

Requires the existing local COCO-Search18 mirror originals and raw TP split1 JSON, the Phase 2 manifest, and the verified CPU perception environment/checkpoints. The report uses the original 24-image internal split, not the official benchmark test set. Source mirror: [CCP_CSHL](https://github.com/RosenholtzLab/CCP_CSHL), commit `401baae571e325eacb000f21783463d2343609ab`.

For each command below supply the same arguments, substituting local paths:

```bash
python scripts/diagnose_spatial.py audit \
  --manifest PATH/manifest.json --source PATH/CCP_CSHL \
  --config configs/pretrained-cpu.yaml \
  --fit reports/phase3/phase3-fit.json --out outputs/phase4a/private
```

Run stages `audit`, `prefix`, `autonomous`, then `finalize`. The autonomous stage persists compressed private predictions so repeated reporting does not rerun inference. `finalize` adds angle and window diagnostics and exports image-level uncertainty without refitting. Trials are keyed by image + participant + task; seeds and prefix steps are averaged inside trials, tasks inside observer/scene, observers inside scenes, then scenes equally. The fixed protocol is [phase4a-protocol.json](phase4a-protocol.json).

```bash
python scripts/report_spatial.py outputs/phase4a/private outputs/phase4a/report \
  --private-overlays outputs/phase4a/private/overlays
```

Open `outputs/phase4a/report/phase4a-report.html` for aggregate charts and tables. Open `outputs/phase4a/private/overlays/index.html` for the 24 original-photograph overlays, individual human paths, frozen prefix candidates and ground-truth target boxes. These private images/events are deliberately excluded from published artifacts. The hosted aggregate report includes per-image human/model density overlays and can be inspected without downloads.

## Evaluation boundaries

- Candidate proposals are frozen after processing only observed preceding fixations, before looking up the next coordinate. Uniform, random, multiscale and dense candidate baselines are evaluation only. Their settings are fixed, not selected on diagnostic results.
- The candidate-location oracle is explicitly scored separately: adding the future point with inherited utility tests coverage without supplying a privileged score. It never enters an autonomous scanpath.
- Target boxes from the original raw records and a full-image SSDLite reference pass are evaluation labels only. Full-image reference detections are not ground truth; outside-reference areas are unclassified rather than verified empty. Complete object masks remain a required follow-up.
- Angular metrics use flat-display bearing separation derived from the authors' rounded 54° × 35° field at 47 cm, not a guessed 16 px/degree or the observer's pseudo-camera. [Published supplement](https://bpb-us-e1.wpmucdn.com/you.stonybrook.edu/dist/b/2177/files/2022/11/CYASHZ-2021-SR-Supplemental.pdf). Approximate geometry does not replace participant-specific calibration.
- The nominal initial event is not always at screen center. Both human and model use the same recorded initial location. Without the raw recording its event provenance cannot be verified further.
- `Σ fixation duration` is not wall-clock viewing time. The legacy RT convention is retained and reported, with lower-window and ordinal-prefix sensitivities. No fabricated saccade gaps or post-response onset correction is made.
- Image bootstrap intervals use 2,000 resamples of six internal test scenes; seeds add no independent scenes. No optimization, new architecture, or population superiority claim follows from these diagnostics.

Recommended next validation data are COCO-FreeView for exploration on the same image family, OSIE for object/semantic allocation, MIT1003 for independent spatial checks, and official COCO-Search18 for a separately specified goal-directed search task. Fixed-image viewing is not interchangeable with active environment learning or head/camera control.

## Verification

66 tests pass in both the core and verified CPU vision environments. Browser checks cover the report, six per-image panels, aggregate downloads, mobile layout and retention of the prior timing lab. The integrity ledger verifies 54 existing source/configuration/report files against the Phase 3 baseline.
