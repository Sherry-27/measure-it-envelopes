# Objectives

These targets were set before capture. MO-2 is met. MO-1 remains unverified without matched ruler readings.

| Field | MO-1 · Physical dimensions | MO-2 · Mask overlap |
| --- | --- | --- |
| What | Mean absolute error in millimetres, separately for the long and short sides. | Mean foreground mask intersection-over-union. |
| How | Compare a fresh photograph of each of ten ruler-measured envelopes with its hand reading. Record refusals separately. Show every row, mean, median, worst absolute error and mean percentage error. | Run the selected checkpoint and frozen threshold on the seven reserved group-10 images. Compare its selected envelope mask to the human outline at the same resolution. Count a miss as zero. |
| Target | Both means at or below 2 mm, with valid measurements for all ten envelopes. This tolerance is meaningful against a millimetre ruler and would distinguish dimensions roughly 5 mm apart. | At least 0.85 over the seven photos. At this level most envelope pixels must agree. An offset edge may still fail the physical target. |
| Conditions | iPhone 11 Pro at locked settings, full envelope visible against a contrasting background and a measured square reference in the same plane. Width is the long side, height the short side. | Fixed test files, one envelope per frame, same calibration and preprocessing as training. Pick checkpoint and score cutoff on validation only. |
| Not included | Bent envelopes, thickness, and the edge stress test. Preserve all refusals in the denominator. | Other envelopes, multiple overlapping targets and the fresh physical accuracy photos. |

## Evidence to collect

- Camera matrix, distortion coefficients, calibration frame count and reprojection RMS in pixels.
- Test mask IoU, mask AP at IoU 0.50, mask AP averaged from 0.50 to 0.95, and precision and recall with operating thresholds stated.
- Paired raw and undistorted physical results with per-envelope ground truth, units, acquisition conditions, counts and errors. If reporting time, take five runs and give the median and range.
- CPU, GPU, RAM, OS, camera, resolution and ruler division. Record local inference and training environments separately.
- Actual outputs and the configuration, checkpoint hash and commands that produced them.

## Final run on the iPhone captures

The photo batch contains 70 distinct HEIC files. The full-frame checkerboard in twenty training exposures makes the earlier clipboard limitation inapplicable to this run. In sensor coordinates at 1200 × 900, the single-radial-coefficient calibration gives RMS **1.035 px**. Leaving each view out, predicting alternate corners, and then swapping them gave a mean RMS of **1.048 px corrected** and **1.163 px with zero distortion**. The board covered a limited part of the image. Physical verification of the nominal 18 mm square remains open. The 36 × 36 mm outer marker dimensions were reported from the user's ruler. [Calibration parameters](../calibration/parameters.json) · [Raw validation results](../calibration/evidence/fullframe-validation.json)

### MO-2 · Segmentation

| Held-out result | Value |
| --- | ---: |
| Mean mask IoU | 0.9291 |
| Mask AP at 0.50 | 1.0000 |
| Mask AP from 0.50 to 0.95 | 0.8920 |
| Precision and recall at mask IoU 0.50 | 1.000 and 1.000 |
| Confidence cutoff | 0.80 |

**MO-2 met.** All seven held-out glass-surface captures were detected. The target was 0.85 mean IoU. These are closely related views on one surface. The result cannot establish performance across manufacturers, cameras or physical envelope sizes. [Per-image results](../models/evidence/test-metrics.json) · [Raw COCO output](../models/evidence/coco-output.txt)

The laptop ran five epochs. Epoch 5 was selected at mean validation IoU 0.9576 on the 13 eligible validation images. One glass-surface validation image was excluded from selection. The test was evaluated after checkpoint selection. [Training log](../models/evidence/training.jsonl) · [Hardware](../models/evidence/environment.json)

### MO-1 · Physical dimensions

**Unverified.** No matched ruler readings for ten identified envelope instances were supplied. Therefore the physical mean absolute error, mean percentage error, median error and worst error remain unknown.

For `IMG_4318.HEIC` the raw command gave **205.6 × 86.4 mm**. After correction it gave **159.7 × 70.7 mm** at confidence **0.9902**. The green outline missed part of the left floral edge. A high detection score can coexist with a biased dimension. Lens correction changed the estimated long side by **-45.9 mm** on this image. Physical error change needs its ruler reading. [Corrected output](../models/evidence/example-inference.json) · [Raw output](../models/evidence/example-raw.json)

The paired results table stores every original in both modes, explicit refusal reasons, and measurements of the reviewed outline using the same marker. The outline comparison checks segmentation geometry only. On the seven held-out captures, 7/7 corrected results and 7/7 raw results were accepted. [70 paired rows](../measurement/evidence/capture-results.csv) · [Split summaries](../measurement/evidence/capture-summary.json)

### Operating point and time

At the selected validation threshold 0.8, 0/13 images were refused in the model-selection check. The raw threshold sweep is saved per epoch. The camera command also requires an unambiguous marker and an outline clear of the image boundary. A covered-marker control returned `reference_not_found`; the marker-only control returned `object_not_found`. These controls are digital edits of an actual validation image. [Control outputs](../measurement/evidence/negative-controls.json)

Five warm CPU runs of `IMG_4298.HEIC` took a median of **11.08 s**, with range **8.76–31.64 s**. They include HEIC decoding through prediction and exclude model loading and overlay writing. Timing is descriptive. The predeclared additional target was mask IoU.
