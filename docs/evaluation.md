# Evaluation contracts and limits

Use your licensed, locally available dataset. The project does not redistribute eye tracking or assume a dataset is accessible.

## Generic manifest

Paths are relative to the manifest. Coordinates are zero-indexed display-image pixels. Normalized coordinates may instead be declared globally; x and y must be in [0,1), never silently clipped. Durations are milliseconds. Images must already represent the exact coordinate frame of the fixations.

```json
{
  "schema_version": 1,
  "dataset_name": "my-audited-eye-tracking-dataset",
  "coordinate_units": "pixels",
  "initial_fixation_included": true,
  "geometry_notes": "Describe resizing, padding, crop and display geometry",
  "records": [{
    "image_id": "scene-001", "image": "images/scene-001.png",
    "width": 640, "height": 480, "split": "test",
    "participant": "P01", "task": "",
    "depth": "depth/scene-001.npz", "depth_kind": "inverse_depth",
    "fixations": [
      {"x": 320, "y": 240, "duration_ms": 200},
      {"x": 150, "y": 180, "duration_ms": 300}
    ]
  }]
}
```

Omit `depth` for a no-depth dataset. A nonempty task is a target label, not a ground-truth target location. No target boxes or next-fixation labels reach the policy. If `initial_fixation_included` is true, only the first human coordinate supplies the starting point and is excluded from metrics. Otherwise the model starts at image center and all human fixations are scored. Fixation duration/count budgets are independent configured values, not copied from each human trial.

Image IDs and decoded-pixel SHA256 values are checked across splits. Select independent images before tuning weights; reserve the test set. `evaluate` refuses training-split evaluation. Participant summaries are reported separately. Correlated trials/seeds do not establish statistical significance. This V1 reports descriptive metrics, not population p-values or confidence intervals.

## COCO-Search18 conversion

`active-observer import-coco RAW_FIXATIONS.json IMAGE_METADATA.json OUTPUT_MANIFEST.json`

The metadata file declares audited images and their display coordinate frame:

```json
{
  "coordinate_units": "pixels", "initial_fixation_included": true,
  "geometry_notes": "Images reconstructed in the recorded 1680x1050 display frame; describe exact padding",
  "images": [{"name": "000000123456.jpg", "image": "images/000000123456.png",
              "width": 1680, "height": 1050, "split": "test"}]
}
```

The converter copies task, subject, X/Y/T and image mapping, discarding target bounding boxes. The generic loader then validates coordinates and dimensions. You must audit correct-trial filtering, absent/present conditions and official splits externally; the converter does not invent them or silently remove bad records. The resulting relative paths must resolve from the output manifest's directory.

## Metrics and limitations

- NSS: standardized generated dwell density at human fixations; undefined for constant maps.
- AUC: tie-correct ranks comparing fixated pixels against all nonfixated pixels, a deterministic comparable AUC; not called AUC-Judd.
- Heatmap correlation: generated/human duration-weighted Gaussian maps.
- Sequence distance: custom DTW normalized by image diagonal and path lengths, not MultiMatch.
- Refixation: a return within 4% of image width to an earlier nonadjacent fixation; adjacent stays do not count as returns.
- Counts, durations and displacements: reported alongside human distributions through per-trial CSVs; amplitudes are pixels unless geometry is separately calibrated.

The fixed Gaussian width is 2.5% of image width. Sensitivity to blur, grid density and foveal size must be tested before scientific conclusions. Monocular depth has no truth labels in eye-tracking data: behavioral improvement does not verify physical depth accuracy. No significance claims should be made from the supplied synthetic fixture.
