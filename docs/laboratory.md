# Recorded visualization laboratory

## Results

- Phase A gate: 35 tests passed before 3D rendering was implemented.
- Final Python suite: 37 passed.
- Browser: 39 Python/JavaScript clock snapshots matched across seven cases.
  Additional checks passed for actual 3D picking/corner coordinates, stored
  depth/quality/validity values, comparison, controls, downloads, orbit/zoom,
  mobile layout and no external requests or JavaScript errors.
- Real Porto run: existing verified SSDLite + MiDaS, 6 fixations, 1.8 seconds
  simulated observation, administrative timeout. Five object hypotheses contain
  six independent geometry records. These are estimates, not corrected labels
  or validated 3D.
- A second pretrained run has depth explicitly disabled. Six labeled mocks
  cover one object, many objects, incompatible scales, empty detections and
  early timeout. Their outputs are not pretrained-performance evidence.

## Precise walkthrough

1. Open real-depth/inspector.html. At t=0, memory, depth and next choice have not
   committed; the actual starting sensory input is visible.
2. Press Next. Depth arrays, objects and stopping commit at the recorded offset.
3. Select a known 2D box, dropdown ID or 3D rectangle. The same stable object ID,
   perceptual uncertainty and independent geometry history appear in the inspector.
4. Orbit/zoom. Fit hypotheses focuses observed support; Show camera context
   includes the frustum. Unknown thickness and orientation stay null in memory.
5. Choose All committed frames, separated. Groups have separate local units;
   spacing is layout, not a reconstructed world.
6. Switch depth channels and hover pixels for exact stored values. The fixed
   depth/quality range is [0,1]. Invalid pixels contain no depth evidence.
7. Inspect signed priority terms or the derived predicted-gain diagnostic.
   Only primitive terms sum to total; per-frame color scaling is labeled.
8. Scrub into an unfinished fixation. New depth/memory/decisions disappear; the
   previous committed map is dated. During saccades, gaze interpolates and the
   input panel labels the retained frame.
9. Compare before/after the last committed fixation. No uncommitted future is
   exposed. Compare predicted gain and measured 2D displacement separately.
   Missing matched measurements are gaps, not invented zeros.
10. At the real run's end, inspect Administrative timeout: no cognitive or
    geometric completion is claimed.

## Browser reproduction

Generate examples and serve outputs/laboratory, then run tests/browser_lab.cjs.
The optional test uses installed Playwright/Chromium. Environment selectors:
PLAYWRIGHT_MODULE, CHROMIUM_PATH, OBSERVER_LAB_URL, OBSERVER_REPLAY_ORACLES.
The default oracle file is work/replay-oracles.json. Build it in Python:

    import json
    from pathlib import Path
    from active_observer.simulation.replay import inspect_at
    names = ["real-depth", "mock-many", "mock-independent-scales", "mock-empty",
             "real-no-depth", "mock-early-timeout", "mock-saccade-timeout"]
    keys = ["elapsed_ms", "fixation_index", "committed_index", "phase",
            "sensory_available", "gaze", "memory", "stopping", "next_choice", "termination"]
    oracles = {}
    for name in names:
        result = json.loads((Path("outputs/laboratory")/name/"run.json").read_text())
        f = result["fixations"][0]
        times = [0, f["offset_ms"]/2, f["offset_ms"]]
        if f["saccade_duration_ms"]:
            times.append(f["offset_ms"]+f["saccade_duration_ms"]/2)
        if len(result["fixations"]) > 1:
            times.append(result["fixations"][1]["onset_ms"]+1)
        times.append(result["duration_ms"])
        oracles[name] = []
        for time in times:
            view = inspect_at(result, time)
            oracles[name].append({"time":time, "expected":{k:view[k] for k in keys},
                "depth_frame_id":view["depth"].get("frame_id") if view["depth"] else None})
    Path("work").mkdir(exist_ok=True)
    Path("work/replay-oracles.json").write_text(json.dumps(oracles))

QA screenshots: outputs/laboratory/desktop.png and mobile.png. Standalone HTML
was tested over loopback HTTP with zero external requests. Managed browser
policy blocks file URLs, so direct file opening was not verified in that runtime.
All scripts and data are embedded; ordinary browsers that permit file URLs
require no local file fetch or CDN.

## Scientific limits

No cross-fixation scale alignment, metric depth, hidden thickness, object
orientation, completed cuboids, ensemble object counts or actual reference-based
3D improvement is available. Faint rectangles encode heuristic uncertainty, not
a confidence interval. Detector mistakes and crop effects are visible as actual
hypotheses. Association is a gated 2D heuristic, not guaranteed correspondence.
Stopping is uncalibrated. Raw depth can show foveation artifacts; it is not
smoothed or corrected for presentation. No full statistical evaluation was run.

## Source changes

Exact paths are listed in changed-files.txt. The schema, scale/storage contract
and repair rationale are in repair-audit.md. Three.js and its MIT license are
vendored in active_observer/viz/vendor.
