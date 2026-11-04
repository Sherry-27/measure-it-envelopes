# Objectives

These targets were set before capture. Results are pending.

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
