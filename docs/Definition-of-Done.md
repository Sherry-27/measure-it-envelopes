# Definition of Done

Leave each item open until an artifact or run demonstrates it. At closeout put the result and its path or commit on the same line. State the reason for any open item.

- [x] Plan and fixed test list precede implementation in Git history. [First commit, 93ee88b](https://github.com/Sherry-27/measure-it-envelopes/commit/93ee88bd57d0a59c295805fcc32b64daf2cf83b6) contains three documents and the 49/14/7 file lists.
- [ ] At least 20 usable board views, measured square size, saved intrinsics, reprojection RMS and a visual check.
- [ ] At least 70 original object photos, with reviewed masks on undistorted images and committed 70/20/10 lists. Check no duplicates across splits.
- [ ] Model trained on the training split; validation selection, config, versions, seed, weights hash and five test metrics recorded.
- [ ] Fresh image produces an overlay, two dimensions in mm and confidence. Missing envelope and missing reference have distinct outputs.
- [ ] At least ten paired hand and system measurements. Raw and corrected outputs, summaries and refusals are traceable to the original files.
- [ ] MO-1 and MO-2 each have a measured result, including misses or refusals.
- [ ] If reached, edge stress levels, refusal threshold cost and one decision with its trade-off are documented.
- [ ] README reproduces the commands on the stated hardware and describes the dataset. Three assessment documents are present.
- [ ] Public pull request and accessible weights checked. Loom is at most five minutes and shows the fresh capture, ruler check, code path and four spoken answers.
- [ ] Unfinished work and limitations listed with reasons.

## Evidence from the board batch

- [x] All 20 received PNGs preserved. The 7 × 9 inner grid was detected in each one. [Per-photo checks and hashes](../calibration/evidence/views.csv). Marked previews are stored locally.
- [ ] Transferable camera parameters and lens-calibration reprojection RMS. The supplied images have different sizes and aspect ratios, with unknown transforms back to the sensor. The board images are the only physical inputs received. Envelope training, physical accuracy, refusal costs and the live demonstration remain unverified.
