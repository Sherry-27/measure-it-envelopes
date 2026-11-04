# Measure It

Measure the long and short edges of one flat paper envelope using an iPhone 11 Pro and a printed reference. Width means the longer side even when the envelope is rotated.

**Current state:** the capture plan, print target and software are prepared. Camera calibration, reviewed labels, trained weights, physical measurements, assessment scores and the Loom depend on the original captures. Only enter measured results after a run.

## Capture

1. Print [the A4 sheet](calibration/print-a4.pdf) at 100%. Cut along the dashed line and mount the checkerboard flat. Keep a white border around the smaller marker. Measure its outer black width and height separately. For the checkerboard, measure several consecutive squares and divide by the count; also record the measuring tool. If using only a ruler, record that the requested calliper check has not been performed.
2. Use the iPhone 11 Pro rear 1× lens in ordinary Photo mode. Choose JPEG, one resolution and one landscape orientation for the whole session. Disable flash and lens switching if applicable. Lock focus at the working distance. Verify the lock before each set; if it cannot persist, use a camera app with fixed manual focus. Retain original files, without cropping, messaging-app compression or digital zoom.
3. Capture about 25 sharp board images. Keep the entire grid in frame. Move the board across the centre and four corners, and rotate or tilt it modestly in both axes. Use the same focus as the envelope photos. Put JPEGs in `calibration/raw/`. Inspect the corrected preview before collecting the dataset.
4. Use ten envelopes across ten independently staged sessions. Shoot seven separate arrangements per envelope. One unaddressed envelope and the small reference should be fully visible in each image, lying flat at the same height. Vary position, orientation, surface and lighting while keeping the camera settings. A new arrangement means physically moving the objects or camera between shots. Group 01–07 is train, 08–09 validation, and 10 test. Name originals as listed in `dataset/splits/`, e.g. `envelope01_01.jpg`, and put them in `dataset/raw/`. Preserve a private copy of the originals. Check for identifying information or GPS tags before publishing.
5. Separately take ten fresh physical-check images, one per envelope. Measure both sides with the ruler, write down the smallest division and fill the CSV template. Also keep one missing-envelope photo, one missing-reference photo, and a fresh envelope for the live demo.

## Setup

Use Python 3.12 and run the commands from the repository root. Make a virtual environment, activate it, then run:

```sh
python -m pip install -r requirements.txt
```

For a Colab GPU, open [the training launcher](https://colab.research.google.com/github/Sherry-27/measure-it-envelopes/blob/dev-test01/models/colab.ipynb) once the real undistorted JPEGs and reviewed masks are on the development branch. The launcher checks the CUDA runtime, records the Git revision and environment, runs the same two project commands, and downloads the checkpoint and evidence together. Use the same versioned project commands there. To select the CPU wheels on Windows, first install the pinned torch and torchvision versions from the PyTorch CPU index, then install the requirements. Record the actual training GPU, CUDA version, OS and library freeze in the run evidence. The intended demo computer has an Intel Core i5-4300U, about 8 GB RAM and Intel HD Graphics on Windows. Exact capture resolution and ruler division are pending collection.

## Commands

Replace the uppercase measurement placeholders with the actual millimetres read from the printed page.

```sh
python -m calibration.camera calibration/raw --square-mm SQUARE_MM
python -m dataset.prepare undistort
```

Inspect the before/after preview and the per-view RMS values in `calibration/parameters.json`. Create a CVAT image task with the undistorted JPEGs in `dataset/images/`. Draw one polygon labelled `envelope` around each outside edge. Review the outlines and export CVAT for images XML.

```sh
python -m dataset.prepare import-cvat annotations.xml
python -m models.train --device cuda --out models/runs/first
python -m models.evaluate --weights models/runs/first/best.pth --out models/evidence
python -m inference.measure new.jpg --weights models/runs/first/best.pth --marker-width-mm WIDTH_MM --marker-height-mm HEIGHT_MM --out overlay.jpg
python -m measurement.compare --table measurement/hand-readings.csv --photos measurement/photos --marker-width-mm WIDTH_MM --marker-height-mm HEIGHT_MM
```

The measurement command writes JSON and an overlay. An `ok` result contains physical width and height. Refusals contain a status and null dimensions. Confidence is the detector score; it does not estimate the physical measurement uncertainty. No scale can be recovered without the measured marker.

## Method and evidence

- The original pixel array is read in sensor orientation, without automatic display rotation. Resolution must match the calibration file. OpenCV lens correction keeps the original camera matrix and image size.
- The model is TorchVision Mask R-CNN with a ResNet-50 FPN backbone. Its instance masks and confidence scores support both outline measurement and detection metrics. The two-class heads are fine-tuned on the envelope training split. Images have a working long edge of at most 1024 pixels to bound CPU memory. Validation selects the checkpoint and operating threshold. Keep the test images sealed until this selection is complete.
- Segmentation AP uses the COCO evaluator and all candidate detections. The operating-point metrics and mean IoU use a single confident prediction. Multiple confident predictions or a miss yield no accepted prediction. Report the seven test images individually; all are views of one held-out envelope.
- The metric conversion is in `inference/geometry.py`, function `metric_box`. The four detected marker corners define a homography into its measured plane. Transform the **outline** first, then fit a rotated rectangle in that plane. A raised or folded envelope violates this assumption.
- `measurement.compare` changes only lens correction between the two runs on each original. The summaries include the full set, refusal counts and the common accepted subset, so differing coverage is visible.
- Put actual calibration, model and accuracy evidence in [Objectives](docs/Objectives.md) and verify each [Definition of Done](docs/Definition-of-Done.md) item. Record deviations in [Plan](docs/Plan.md). Release the chosen weights and add the download link and SHA-256 here after training. On Windows, use `Get-FileHash models/runs/first/best.pth -Algorithm SHA256`.

