# Measure It | Plan

The first version of this plan and the 49/14/7 filename lists were committed before code. The goal is to measure the long and short sides of a flat envelope using an iPhone 11 Pro, a printed reference and a ruler.

| Active minutes | Intended work |
| ---: | --- |
| 25 | Commit the plan, objectives, checklist and split lists. |
| 55 | Lock the camera, capture about 25 board poses, fit at least 20 usable views and check straight edges. |
| 110 | Capture 70 envelope photos, correct them and review the outer-boundary masks. |
| 65 | Fine-tune pretrained Mask R-CNN. Select the checkpoint and score cutoff on validation. |
| 55 | Run one original through correction, segmentation, reference detection and metric conversion. Check missing inputs. |
| 65 | Compare ten hand-measured instances against raw and corrected predictions. |
| 45 | After the accuracy table, test worsening edge position and record one decision with its cost. |
| 60 | Reproduce the commands, finish evidence and record the five-minute demonstration. |

Total active budget is eight hours. Unattended training and installation waits are separate. Skip a web app, multiple architectures and parameter sweeps. Protect the recording and declare anything unfinished.

## Capture and split updates

Eight physical envelopes were reported available. The initial plan proposed ten sessions with independent envelope identities. The delivered 70 photos use four surfaces and include repeated front/back appearances. Physical identities were initially unknown. After the frozen model run, the capture was confirmed to use the same physical envelope in all 70 photos.

Original camera files were bound to the reserved slots before fine-tuning. Training uses the twenty board-and-envelope captures plus 29 black-surface images. The validation set has five white, eight marble and one glass view. The glass view is excluded from checkpoint and threshold selection. The seven other glass views form the test. The test therefore covers a narrow scene.

The earlier twenty clipboard images had different crop sizes. Once the native HEIC files arrived, their orientation metadata recovered the common sensor frame. Calibration used the twenty full-frame training exposures. No test image was used to fit the camera or choose the checkpoint.

## Decision record

**Camera fit.** A free OpenCV fit reached 0.837 px RMS but estimated a focal length far from the phone's EXIF prior and large higher-order coefficients. The deployed model fixes the centre, square pixels and focal prior, then fits one radial coefficient. It costs 0.198 px of training RMS. It improved 19 of 20 omitted-view checks over zero distortion. A wider set of board positions would allow this choice to be revisited. The later ruler reading confirmed 90 mm over five squares in each direction, or 18 mm per square. The same 1 mm-division ruler measured the envelope as 160 × 70 mm. The 36 mm reference had been measured earlier.

**Training.** The Colab plan changed to a local CPU run. ResNet stages were frozen, with a smaller proposal budget and 256–384 px model inputs. The FPN and detection components still fine-tune on the envelope masks. The cost is coarser edges and a slower laptop run. The final model missed part of the floral edge in one held-out image despite high confidence. The T4 notebook retains the same code for reproduction.

**Ruler comparison.** Once the physical identity and dimensions were supplied, all 70 original predictions were scored. They are repeated views of one specimen. The original ten-envelope objective is not met, and the physical error exceeds the 2 mm target. The failed target is retained.

**Crop decision.** A requested automatic crop refinement was tested on the 13 eligible validation views, with a fixed 20% padding rule and restored camera coordinates. It worsened long/short MAE from 9.94/2.36 mm to 13.95/6.68 mm. It remains optional and off by default. Rejecting it preserves the coarser baseline boundaries but avoids the measured degradation and a second inference pass.

**Failure-test change.** The planned edge-position reshoot was replaced by a digital brightness sweep after the ruler table was complete. This isolates pixel dimming on two fixed validation views and demonstrates increasing refusals. Its cost is no evidence about real lighting changes, camera exposure adaptation or glare.

**Remaining scope.** There are no ten distinct measured envelopes, no controlled physical lighting reshoot, and no calliper reading. These limits are declared. The candidate still supplies the fresh-photo recording and recruiter reply.
