# Measure It

Measure a flat envelope from one iPhone 11 Pro photograph. Place the printed marker beside the envelope on the same plane and keep the full black border visible. Width is the longer side of the fitted envelope rectangle. Height is the shorter side.

## Run a photograph

Use Python 3.12. The measured outer border of this marker is **36 × 36 mm**. Print files, camera parameters, configuration and evidence are on `dev-test01`. Install the requirements, put the selected checkpoint at `models/runs/first/best.pth`, then run from the repository root.

```sh
python -m pip install -r requirements.txt
python -m inference.measure my-envelope.HEIC --out measured.jpg
```

The command prints JSON and saves an overlay. It accepts native iPhone HEIC or a JPEG in the documented sensor frame. Arbitrary screenshots or crops do not have a known calibration. The program returns null dimensions and an explicit reason if the envelope or reference cannot be measured. The score is the instance detector's confidence. Millimetre uncertainty requires a separate physical comparison.

## Photos and camera

The capture contains **70 distinct HEIC files** from the same 4.25 mm iPhone 11 Pro rear lens. All have a native 4032 × 3024 sensor frame. Sixty-two use a display rotation. Decoding reverses this rotation and resizes uniformly to 1200 × 900 before lens correction. Twenty full-frame training exposures include the 7 × 9 inner-corner calibration board. Their original camera coordinates supersede the earlier differently cropped exports.

OpenCV found all 1,260 intersections. The deployed camera model fixes square pixels and uses the 26 mm-equivalent EXIF focal length as a prior. It fits one radial coefficient because these board views cover a limited part of the frame. Training-view reprojection RMS is **1.035 px** at 1200 × 900. Omitting one view and predicting alternate corners reduced mean error from 1.163 px to 1.048 px with correction. The printed square size used for extrinsics is the nominal 18 mm. The measured marker sets the scale used for envelope measurements. [Camera parameters](calibration/parameters.json) · [Validation by view](calibration/evidence/fullframe-validation.json)

The 70 object masks came from OpenCV edge and colour proposals, visual inspection of every outline, and twenty manual boundary corrections. The reviewed polygons are in sensor coordinates. The mask and its matching photograph use the same undistortion map. The reference marker and checkerboard are outside the envelope label.

| Split | Photos | Camera scenes |
| --- | ---: | --- |
| Train | 49 | 20 checkerboard-and-envelope views and 29 black-surface views. |
| Validation | 14 | Five white, eight marble and one glass. The glass exposure is excluded from checkpoint and threshold selection. |
| Test | 7 | The seven glass-surface views of the horizontally placed envelope. |

The file lists were committed first. The original filenames and hashes were bound to those lists before fine-tuning. The supplied photos did not follow the proposed ten independent object sessions. Physical envelope identities could not be verified. The test is one tabletop scene with similar views, which limits generalization claims. [Capture index](dataset/capture-index.csv) · [Reviewed outlines](dataset/reviewed-outlines.json)

## Reproduce

The private delivery includes `Envelope-Dataset.zip`. Extract it in the repository root to populate `dataset/images` and `dataset/masks`. Alternatively, rebuild both from a directory of the original HEIC files. The rebuild checks their hashes. The restored 70 calibrated images and 70 masks were compared byte-for-byte with the training copies.

```sh
python -m calibration.fit_iphone ORIGINAL_HEIC_FOLDER
python -m dataset.restore ORIGINAL_HEIC_FOLDER
python -m models.train --device cpu --out models/runs/reproduction
python -m models.evaluate --device cpu --weights models/runs/reproduction/best.pth --out models/reproduction-evidence
python -m measurement.audit_outputs ORIGINAL_HEIC_FOLDER --weights models/runs/reproduction/best.pth --out measurement/reproduction-evidence
```

The versioned training configuration fine-tunes TorchVision Mask R-CNN with a COCO-pretrained ResNet-50/FPN and new two-class heads. The ResNet stages are frozen. The FPN, proposal and detection components can learn from the envelope masks. This run uses seed 2026, batch size 1, two CPU threads and right-angle rotations and horizontal flips on training data only. Validation selects the epoch and operating threshold. The CPU run uses an Intel Core i5-4300U, Windows 10 and about 8 GB RAM. [Run environment](models/evidence/environment.json) · [Training configuration](models/config.json)

**Colab is optional.** The T4 notebook asks for the dataset ZIP, invokes the same training and evaluation modules on CUDA, and downloads its own run bundle. The recorded assessment scores come from the local CPU run. [Open the Colab notebook](https://colab.research.google.com/github/Sherry-27/measure-it-envelopes/blob/dev-test01/models/colab.ipynb)

## Geometry and physical check

`inference/measure.py` corrects the photo before segmentation and marker detection. `inference/geometry.py` maps the predicted contour from pixels into the measured marker plane and fits the rotated rectangle there. A folded envelope, raised marker or a marker with a bent border can invalidate the conversion.

For actual physical accuracy, measure each independently identified envelope with a ruler or calliper, place its longer and shorter readings in the hand-readings CSV, and pair each row with the correct original. The comparison saves both raw and corrected predictions, per-instance errors and refusals.

```sh
python -m measurement.compare --table measurement/hand-readings.csv --photos measurement/photos
```

[Plan and decisions](docs/Plan.md) · [Objectives and results](docs/Objectives.md) · [Definition of Done](docs/Definition-of-Done.md)

