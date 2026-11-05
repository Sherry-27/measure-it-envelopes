"""Measure errors and refusals under a fixed digital brightness sweep.

This edits pixel intensity in two existing validation captures. It is not a
physical lighting experiment and does not add images to the training dataset.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.camera import read_image
from inference.measure import Meter
from measurement.compare import summarize


SOURCES = ("IMG_4298.HEIC", "IMG_4308.HEIC")
FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def run(originals, out, weights):
    readings = json.loads(Path("measurement/physical-readings.json").read_text(encoding="utf-8"))
    envelope = readings["envelope"]
    if not envelope["all_70_photos_same_physical_envelope"]:
        raise ValueError("This diagnostic needs confirmed dimensions for both source photos.")
    baseline = json.loads(Path("measurement/evidence/physical/summary.json").read_text(encoding="utf-8"))
    if baseline["raw"]["total"] < 10:
        raise ValueError("Complete the ruler comparison before the failure-condition diagnostic.")
    index = {r["source"]: r for r in csv.DictReader(Path("dataset/capture-index.csv").open(encoding="utf-8"))}
    assert all(index[n]["split"] == "val" and index[n]["use_for_selection"] == "1" for n in SOURCES)
    meter = Meter(Path("calibration/parameters.json"), weights, 36., 36., "cpu")
    out.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for source in SOURCES:
        original = read_image(originals/source)
        for factor in FACTORS:
            darker = np.rint(original.astype(np.float32)*factor).astype(np.uint8)
            prediction, overlay = meter.measure(darker)
            row = dict(source=source, specimen_id=envelope["specimen_id"], split="val", intensity_factor=factor,
                       hand_width_mm=envelope["long_side_mm"], hand_height_mm=envelope["short_side_mm"])
            for key in ("status", "confidence", "width_mm", "height_mm"):
                row["corrected_"+key] = prediction[key]
            for axis in ("width", "height"):
                value = prediction[axis+"_mm"]
                row[axis+"_abs_error_mm"] = None if value is None else abs(value-row["hand_"+axis+"_mm"])
            rows.append(row)
            tile = np.full((340,400,3),248,np.uint8)
            tile[:300] = cv2.resize(overlay,(400,300),interpolation=cv2.INTER_AREA)
            label = f"{source[:-5]}  intensity x{factor:g}  {prediction['status']}"
            cv2.putText(tile,label,(6,315),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1,cv2.LINE_AA)
            if prediction["status"] == "ok":
                label = f"Abs error: {row['width_abs_error_mm']:.2f} / {row['height_abs_error_mm']:.2f} mm"
            else:
                label = "No dimension reported"
            cv2.putText(tile,label,(6,332),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
            tiles.append(tile)
            print(json.dumps(row,allow_nan=False),flush=True)
    summaries=[]
    for factor in FACTORS:
        group=[r for r in rows if r["intensity_factor"]==factor]
        summaries.append(dict(intensity_factor=factor,**summarize(group,"corrected")))
    summary=dict(
        study="digital brightness sensitivity; not a physical illumination experiment",
        operation="round(sensor-frame RGB * intensity_factor), clipped to uint8; no geometry changes",
        sources=list(SOURCES),factors=list(FACTORS),independent_physical_specimens=1,
        choice="Lighting is varied in software while each view, marker, outline geometry and physical dimensions stay fixed.",
        selection="First capture from each of the white and marble validation surfaces; factors fixed before this run.",
        interpretation="One deterministic run per source and factor; repeated pixels are not independent samples.",
        limitations=["Global digital dimming omits camera noise, exposure adaptation, local shadows and glare.",
                     "Both source views are from the existing validation data; this is a diagnostic after model freezing.",
                     "No threshold or checkpoint was changed using these outcomes."],
        by_factor=summaries,
        hand_readings_sha256=hashlib.sha256(Path("measurement/physical-readings.json").read_bytes()).hexdigest(),
        checkpoint_sha256=json.loads(Path("models/evidence/checkpoint.json").read_text(encoding="utf-8"))["sha256"])
    with (out/"trials.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    canvas=cv2.vconcat([cv2.hconcat(tiles[i:i+3]) for i in range(0,len(tiles),3)])
    header=np.full((55,canvas.shape[1],3),248,np.uint8)
    cv2.putText(header,"DIGITAL DIMMING TEST - real model outputs, reported envelope 160 x 70 mm",
                (12,24),cv2.FONT_HERSHEY_SIMPLEX,.62,(20,20,20),1,cv2.LINE_AA)
    cv2.putText(header,"Two existing validation photos; these are simulated intensities, not new camera captures.",
                (12,46),cv2.FONT_HERSHEY_SIMPLEX,.53,(20,20,20),1,cv2.LINE_AA)
    cv2.imwrite(str(out/"preview.jpg"),cv2.vconcat([header,canvas]))
    print(json.dumps(summary,allow_nan=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("originals",type=Path)
    parser.add_argument("--out",type=Path,default=Path("measurement/evidence/brightness"))
    parser.add_argument("--weights",type=Path,default=Path("models/runs/first/best.pth"))
    a=parser.parse_args()
    run(a.originals,a.out,a.weights)

