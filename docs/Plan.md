# Measure It | Plan

Measure the long and short sides of a flat paper envelope in millimetres. Use ten physical envelopes, an iPhone 11 Pro and a ruler. Record the envelope dimensions, smallest ruler division and capture resolution at collection. Flat envelopes are easy to hold against a ruler. Use unaddressed faces and vary the backgrounds.

| Minutes | Work and stopping point |
| ---: | --- |
| 25 | Commit the plan, objectives, checklist and filename lists. Assign one envelope and one capture session to each group. |
| 55 | Lock the phone settings. Photograph the printed board in about 25 poses. Fit a camera model from at least 20 accepted views and inspect straight edges after correction. |
| 110 | Take seven independent photos of each envelope, moving the envelope and reference between shots. Save originals, undistort, and trace the outer boundary. Manually review the masks. |
| 65 | Fine-tune a pretrained TorchVision Mask R-CNN on the training groups. Select a checkpoint and confidence cutoff on validation only. Run the locked test once after selection. |
| 55 | Build the single-image command. Map the predicted contour onto the plane defined by a measured square marker. Check the overlay and both missing-input cases. |
| 65 | Ruler-measure the ten envelopes individually. Photograph fresh placements. Compare the same images with and without lens correction, saving each result and refusal. |
| 45 | If the accuracy table exists, move one envelope toward the image edge and check error and refusals. Record a design decision and its cost. |
| 60 | Re-run setup and inference, attach evidence, and record a fresh photo and ruler check in the five-minute demo. Open the pull request. |

Total active work budget is eight hours. Install and training waits are separate. Skip a web app, hyperparameter sweeps and multiple architectures. If a stage overruns, protect the demo and document the last verified stage. The edge experiment comes after the accuracy check.

## Fixed capture assignment

Groups 01–07 train, 08–09 validate, and 10 tests. Each group is a different physical envelope and independently staged session. The seven frames within a group vary position, rotation and background. The three lists name future undistorted JPEGs and are fixed before any code or capture. Keep group identities when copying originals. The test covers one unseen envelope; that limits conclusions about other designs.

## Working decisions

Use the reference marker corners to establish a metric plane. The envelope and marker must lie flat at the same height. Use a ruler for the available ground truth and record its resolution. The brief asks for a calliper for the printed square. Record the actual tool used and flag any deviation. Record changes here when they occur.

## Capture update

Eight envelopes are available. The planned ten session IDs and the original split lists stay fixed. Training groups 01 through 07 can reuse the first five physical envelopes. Validation groups 08 and 09 reserve envelopes 6 and 7, and group 10 reserves envelope 8. The physical accuracy check still calls for ten measured instances.

The first board batch contains twenty PNG exports of different sizes and aspect ratios. The original crop offsets and scaling are unavailable. Resizing each export to the same rectangle would move grid points by different unknown amounts. The inspection therefore keeps each photo in its supplied coordinates and saves the corner overlay and per-view result. The cost is that this batch cannot establish the distortion correction to apply to a fresh phone image.
