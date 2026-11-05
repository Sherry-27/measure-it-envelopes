# Objectives

These targets were set before capture. MO-2 is met. MO-1 is not met: the dimensional errors exceed 2 mm, and the delivered photos show one physical envelope rather than the planned ten.

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

The photo batch contains 70 distinct HEIC files. The full-frame checkerboard in twenty training exposures makes the earlier clipboard limitation inapplicable to this run. In sensor coordinates at 1200 × 900, the single-radial-coefficient calibration gives RMS **1.035 px**. Leaving each view out, predicting alternate corners, and then swapping them gave a mean RMS of **1.048 px corrected** and **1.163 px with zero distortion**. The board covered a limited part of the image. The later ruler reading was 90 mm across five consecutive squares in each axis, confirming an average square size of 18 mm. A 1 mm-division ruler was used, rather than the requested calliper. The 36 × 36 mm outer marker dimensions were reported from the user's ruler. [Calibration parameters](../calibration/parameters.json) · [Raw validation results](../calibration/evidence/fullframe-validation.json)

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

**Not met.** After the model was frozen, the envelope was reported as **160 × 70 mm**, and all 70 photos were confirmed to show that same physical envelope. The scale has nine intermediate ticks per centimetre, so its smallest division is **1 mm**. There is one independent specimen and 70 repeated photo trials, including the existing training and validation views. This is narrower than the original objective's fresh photos of ten envelopes. [Reported readings](../measurement/physical-readings.json) · [Matched input table](../measurement/hand-readings.csv)

| Photos / mode | Accepted | Long MAE | Short MAE | Long MAPE | Short MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All / raw | 66/70 | 15.32 mm | 4.77 mm | 9.57% | 6.81% |
| All / corrected | 68/70 | 19.50 mm | 5.06 mm | 12.19% | 7.23% |
| Same 66 / raw | 66/66 | 15.32 mm | 4.77 mm | 9.57% | 6.81% |
| Same 66 / corrected | 66/66 | 19.46 mm | 5.03 mm | 12.17% | 7.19% |
| Test 7 / raw | 7/7 | 23.25 mm | 6.53 mm | 14.53% | 9.33% |
| Test 7 / corrected | 7/7 | 16.32 mm | 4.11 mm | 10.20% | 5.87% |

| Accepted photo trials | Long mean / median / worst absolute error | Short mean / median / worst absolute error |
| --- | --- | --- |
| Raw (66) | 15.32 / 14.86 / 45.61 mm | 4.77 / 3.90 / 17.38 mm |
| Corrected (68) | 19.50 / 18.02 / 121.32 mm | 5.06 / 4.06 / 41.87 mm |

Every result, refusal and per-axis error is retained in the [70-row physical table](../measurement/evidence/physical/instances.csv). The [raw summary](../measurement/evidence/physical/summary.json) also gives train/validation/test groups, signed percentage errors and counts within 2 mm on both axes. Decimal error values are arithmetic against the reported ruler values, not claims of sub-millimetre ground-truth precision. No dimensions were used as inference inputs.

On the **same 66 accepted photos**, correction increased mean absolute error by **4.15 mm long-side** and **0.26 mm short-side**. On the seven reserved test views it reduced the means from **23.25 / 6.53 mm** to **16.32 / 4.11 mm**. Better checker-corner reprojection did not guarantee better object dimensions. The small marker, visible paper curvature, limited calibration coverage and segmentation boundary errors are possible contributors; this dataset does not isolate their separate effects.

For the saved `IMG_4318.HEIC` example, raw output was **205.61 × 86.42 mm**, with absolute errors **45.61 / 16.42 mm**. Corrected output was **159.69 × 70.67 mm**, with errors **0.31 / 0.67 mm**. The predicted mask still misses part of an edge, so this close numerical result does not prove that every geometric step is accurate. The full table is the relevant accuracy evidence.

### Failure condition · Digital dimming

After the ruler comparison, lighting sensitivity was tested by multiplying pixel intensities in the first white-surface and marble-surface validation photos by six fixed factors. Geometry, model and threshold stayed fixed. These are **digital variants of two existing views**, not new captures or a physical illumination experiment.

| Intensity factor | Accepted / attempted | Long-side MAE | Short-side MAE |
| ---: | ---: | ---: | ---: |
| 1 | 2/2 | 15.03 mm | 0.67 mm |
| 0.5 | 1/2 | 16.07 mm | 0.50 mm |
| 0.25 | 1/2 | 16.54 mm | 2.09 mm |
| 0.125 | 0/2 | not reported | not reported |
| 0.0625 | 0/2 | not reported | not reported |
| 0.03125 | 0/2 | not reported | not reported |

For the marble view, long-side error changed from 15.68 mm at full intensity to 16.54 mm at one-quarter intensity. At one-eighth intensity the marker was not found. The white view failed the confidence cutoff already at one-half intensity. Refusals remain in the denominator; an empty accepted set is not zero error.

Lighting was chosen because this test changes one image property while keeping the scene geometry fixed. It does not reproduce exposure adaptation, sensor noise, shadows or glare. A physical reshoot under measured illumination remains outside the delivered evidence. [All 12 outputs](../measurement/evidence/brightness/trials.csv) · [Summary](../measurement/evidence/brightness/summary.json) · [Preview](../measurement/evidence/brightness/preview.jpg)

### Requested crop check

An automatic second segmentation pass used the coarse mask's bounding box with 20% padding per side and a 16 px minimum. It ran **after undistortion**, then mapped the fine mask back to full-frame coordinates. Ground-truth outlines and dimensions did not choose the crop.

On the 13 eligible validation views, long-side MAE worsened from **9.94 to 13.95 mm** and short-side MAE from **2.36 to 6.68 mm**, with 13/13 accepted in both modes. The crop option stays off by default. No test image was processed in this experiment, and the original test metrics still describe the full-frame baseline. [Per-photo comparison](../measurement/evidence/crop/trials.csv) · [Summary](../measurement/evidence/crop/summary.json) · [Preview](../measurement/evidence/crop/preview.jpg)

### Operating point and time

At the selected validation threshold 0.8, 0/13 images were refused in the model-selection check. The raw threshold sweep is saved per epoch. The camera command also requires an unambiguous marker and an outline clear of the image boundary. A covered-marker control returned `reference_not_found`; the marker-only control returned `object_not_found`. These controls are digital edits of an actual validation image. [Control outputs](../measurement/evidence/negative-controls.json)

Five warm CPU runs of `IMG_4298.HEIC` took a median of **11.08 s**, with range **8.76–31.64 s**. They include HEIC decoding through prediction and exclude model loading and overlay writing. Timing is descriptive. The predeclared additional target was mask IoU.
