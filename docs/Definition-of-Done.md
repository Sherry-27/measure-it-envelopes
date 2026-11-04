# Definition of Done

| Assessment item | Status and evidence |
| --- | --- |
| 1. Calibrate the camera | **[x]** 20 full-frame training views with 63 corners each. Saved camera matrix, distortion, per-view errors and 1.035 px RMS. [Before/after preview](../calibration/evidence/fullframe-preview.jpg). **[ ]** Independent calliper reading for the printed square. The 18 mm value is nominal. [Fit](../calibration/parameters.json) |
| 2. Own labelled data | **[x]** 70 source hashes, 70 reviewed masks and 49/14/7 committed slots. All 140 generated image/mask files rebuilt byte-for-byte. The camera originals, prepared dataset and test overlays are linked from the README. [Source mapping](../dataset/capture-index.csv) · [Labels](../dataset/reviewed-outlines.json) |
| 3. Train and evaluate segmentation | **[x]** Five local CPU epochs. Frozen epoch 5. Test IoU 0.929, AP50 1.000, AP50:95 0.892, precision 1.000 and recall 1.000. [Raw scores](../models/evidence/test-metrics.json) |
| 4. One image to millimetres | **[x]** One command ran on a held-out original HEIC and saved dimensions, confidence and an overlay. Negative-control branch checks **[x]**. [Prediction](../models/evidence/example-inference.json) · [Controls](../measurement/evidence/negative-controls.json) |
| 5. Accuracy against a ruler | **[ ]** Ten matched physical-instance readings have not been supplied. The model outputs and comparison script are ready. No physical accuracy claim is made. |
| 6. Effect of undistortion | **[x]** Raw/corrected estimates for 70 images and held-out checker corner comparisons. **[ ]** Paired physical measurement errors require ruler truth. [Paired table](../measurement/evidence/capture-results.csv) |
| 7. Additional measurable target | **[x]** Predeclared mask IoU ≥ 0.85. Observed 0.9291. Five actual CPU timings are also recorded. [Objectives](Objectives.md) |
| 8. Worsening failure condition | **[ ]** No controlled severity series with ruler truth was collected. The plan defers this until item 5 is finished. |
| 9. Refuse unreliable measurements | **[x]** Threshold sweep on validation, selected cutoff 0.80, test refusals and explicit missing-input outcomes. [Threshold costs](../models/evidence/training.jsonl) · [Capture outcomes](../measurement/evidence/capture-summary.json) |
| 10. Decision record | **[x]** Camera-model constraints and CPU trade-offs recorded with alternative fit and cost. [Plan](Plan.md) |

### Reproducibility and submission

- [x] The original [first commit](https://github.com/Sherry-27/measure-it-envelopes/commit/93ee88bd57d0a59c295805fcc32b64daf2cf83b6) contains the plan, checklist, objectives and split files before code.
- [x] Branch `dev-test01`, incremental commits and [pull request to main](https://github.com/Sherry-27/measure-it-envelopes/pull/1).
- [x] The README documents setup, camera, data, hardware, training, reproduction and inference. All three assessment documents are in this directory.
- [x] Selected weights published and the SHA-256 confirmed against the release asset. [Checkpoint metadata](../models/evidence/checkpoint.json)
- [ ] Candidate's Loom ≤ 5 minutes with fresh capture, live ruler check, code walkthrough and all four spoken answers.
- [ ] Candidate sends the PR and accessible Loom links to the recruiter within the assigned window.
