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

## Board observations from the supplied photos

The programme found 63 inner corners in each of 20 PNGs, giving 1,260 observed corner positions. Four views required the fallback detector. The sheet has visible creases. Each image has a different width-height combination and the crop offsets are unknown. The printed label gives a nominal square size of 18 mm. A physical square measurement has not been supplied.

For each view, a planar transform predicts alternate grid intersections from the other intersections. Swapping the two groups lets every corner act as a check. Across these 20 per-photo RMS errors, the mean is 4.46 px, median 4.19 px and worst 9.26 px (board_09.png). This check is in the pixel units of each export and includes effects of paper shape and lens geometry. It does not isolate lens distortion, provide camera-calibration reprojection RMS or measure envelope error. Both assessment objectives remain unmeasured.

[Per-photo table](../calibration/evidence/views.csv) · [Detected corners](../calibration/evidence/corners.json)
