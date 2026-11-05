"""Compare one fixed crop-refinement option on eligible validation images only."""
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


def run(originals, out):
    index=list(csv.DictReader(Path("dataset/capture-index.csv").open(encoding="utf-8")))
    chosen=sorted([r for r in index if r["split"]=="val" and r["use_for_selection"]=="1"],key=lambda r:r["source"])
    assert len(chosen)==13 and all(r["split"]!="test" for r in chosen)
    saved={r["source"]:r for r in csv.DictReader(Path("measurement/evidence/capture-results.csv").open(encoding="utf-8"))}
    truth=json.loads(Path("measurement/physical-readings.json").read_text(encoding="utf-8"))["envelope"]
    meter=Meter(Path("calibration/parameters.json"),Path("models/runs/first/best.pth"),36.,36.,"cpu")
    out.mkdir(exist_ok=True,parents=True)
    rows,tiles=[],[]
    for rec in chosen:
        source=rec["source"]
        row=dict(source=source,split="val",specimen_id=truth["specimen_id"],
                 hand_width_mm=truth["long_side_mm"],hand_height_mm=truth["short_side_mm"])
        for key in ("status","confidence","width_mm","height_mm"):
            v=saved[source]["corrected_"+key]
            row["baseline_"+key]=v if key=="status" else (float(v) if v else None)
        pred,overlay=meter.measure(read_image(originals/source),refine_crop=True)
        for key in ("status","confidence","width_mm","height_mm"):
            row["refined_"+key]=pred[key]
        row["crop_bounds_xyxy"]=json.dumps(pred.get("crop_bounds_xyxy"))
        for mode in ("baseline","refined"):
            for axis in ("width","height"):
                value=row[f"{mode}_{axis}_mm"]
                row[f"{mode}_{axis}_abs_mm"]=None if value is None else abs(value-row["hand_"+axis+"_mm"])
        rows.append(row)
        if source in ("IMG_4298.HEIC","IMG_4308.HEIC","IMG_4313.HEIC","IMG_4317.HEIC"):
            tile=np.full((365,480,3),248,np.uint8)
            tile[:320]=cv2.resize(overlay,(480,320))
            cv2.putText(tile,source+" - automatic crop refinement",(6,337),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
            text=pred["status"] if pred["status"]!="ok" else f"{pred['width_mm']:.2f} x {pred['height_mm']:.2f} mm; truth 160 x 70"
            cv2.putText(tile,text,(6,356),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
            tiles.append(tile)
        print(json.dumps(row,allow_nan=False),flush=True)
    stats={mode:summarize(rows,mode) for mode in ("baseline","refined")}
    common=[r for r in rows if r["baseline_status"]=="ok" and r["refined_status"]=="ok"]
    stats["common_accepted"]={m:summarize(common,m) for m in ("baseline","refined")}
    stats["crop_rule"]="Coarse mask bounding box, 20% padding per side with a 16 px minimum; second prediction mapped back into the full corrected frame."
    stats["uses_ground_truth_to_crop"]=False
    stats["test_images_processed"]=0
    stats["independent_physical_specimens"]=1
    a,z=stats["baseline"],stats["refined"]
    stats["validation_favors_crop"]=(z["accepted"]>=a["accepted"] and z["width"] is not None and
                                    z["width"]["mae_mm"]<a["width"]["mae_mm"] and z["height"]["mae_mm"]<=a["height"]["mae_mm"])
    stats["scope"]="Post-training validation diagnostic. Original test scores describe the full-frame baseline only. No unseen-object or new-test-set claim."
    stats["checkpoint_sha256"]=json.loads(Path("models/evidence/checkpoint.json").read_text(encoding="utf-8"))["sha256"]
    with (out/"trials.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if tiles:
        cv2.imwrite(str(out/"preview.jpg"),cv2.vconcat([cv2.hconcat(tiles[:2]),cv2.hconcat(tiles[2:])]))
    print(json.dumps(stats,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("originals",type=Path)
    p.add_argument("--out",type=Path,default=Path("measurement/evidence/crop"))
    a=p.parse_args()
    run(a.originals,a.out)

