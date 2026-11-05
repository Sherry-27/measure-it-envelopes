# Definition of Done

| Assessment item | Status and evidence |
| --- | --- |
| 1. Calibrate the camera | **[x]** 20 full-frame views, 63 corners each, saved parameters and 1.035 px RMS. Five squares measured 90 mm in both axes, confirming 18 mm per square. A 1 mm-division ruler was used in place of the requested calliper. [Fit](../calibration/parameters.json) · [Readings](../measurement/physical-readings.json) |
| 2. Own labelled data | **[x]** 70 original photos, 70 reviewed masks and fixed 49/14/7 slots. Rebuilt image/mask pairs matched byte-for-byte. All photos show one physical envelope; the split tests new views of that same object. [Source mapping](../dataset/capture-index.csv) · [Labels](../dataset/reviewed-outlines.json) |
| 3. Train and evaluate segmentation | **[x]** Five CPU epochs. Test IoU 0.9291, AP50 1.000, AP50:95 0.892, precision/recall 1.000. [Raw scores](../models/evidence/test-metrics.json) |
| 4. One image to millimetres | **[x]** Original HEIC command and desktop photo picker produce dimensions, confidence, overlay and explicit refusals. [Prediction](../models/evidence/example-inference.json) · [Controls](../measurement/evidence/negative-controls.json) |
| 5. Accuracy against a ruler | **[x]** Per-photo table for 70 repeated trials against the reported 160 × 70 mm ruler reading, including mean/median/worst and percentage errors. **[ ]** Original objective of ten independent envelopes. [All rows](../measurement/evidence/physical/instances.csv) · [Summary](../measurement/evidence/physical/summary.json) |
| 6. Effect of undistortion | **[x]** Paired raw/corrected physical errors on the same 66 accepted photos: long-side MAE 15.32 → 19.46 mm; short-side 4.77 → 5.03 mm. The seven test views improved but still missed the target. [Paired evidence](../measurement/evidence/physical/summary.json) |
| 7. Additional measurable target | **[x]** Mask IoU target ≥ 0.85 met at 0.9291. The 2 mm physical target is explicitly missed. [Objectives](Objectives.md) |
| 8. Worsening failure condition | **[x]** Six fixed digital intensity levels on two validation views, with errors/refusals at every level. **[ ]** A controlled physical lighting reshoot. [Outputs](../measurement/evidence/brightness/trials.csv) · [Preview](../measurement/evidence/brightness/preview.jpg) |
| 9. Refuse unreliable measurements | **[x]** Frozen score cutoff 0.80, threshold costs, missing-input controls, and 0/2 accepted at intensity 0.125 or below. High-confidence geometric errors remain a limitation. [Brightness evidence](../measurement/evidence/brightness/summary.json) |
| 10. Decision record | **[x]** Camera constraints, CPU training, rejected crop refinement and digital-test limitations. [Plan](Plan.md) |

## Reproducibility and submission

- [x] The [first commit](https://github.com/Sherry-27/measure-it-envelopes/commit/93ee88bd57d0a59c295805fcc32b64daf2cf83b6) contains the plan, checklist, objectives and split files before code.
- [x] Branch `dev-test01`, incremental commits and [pull request to main](https://github.com/Sherry-27/measure-it-envelopes/pull/1).
- [x] README covers setup, camera, data, hardware, training, inference, ruler scoring and the desktop launcher. Exactly three assessment documents are in this directory.
- [x] Published checkpoint SHA-256 verified. [Weights metadata](../models/evidence/checkpoint.json)
- [x] Requested crop experiment completed on validation only; worse errors kept it off by default. [Evidence](../measurement/evidence/crop/summary.json)
- [ ] Candidate's Loom ≤ 5 minutes with fresh photo, live ruler check, code walkthrough and all four spoken answers.
- [ ] Candidate sends the PR and accessible Loom links to the recruiter within the assigned window.
