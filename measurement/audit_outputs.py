"""Compare measurement outputs on identical captures before and after lens correction."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np

from calibration.camera import correct, read_image
from inference.geometry import find_reference, metric_box
from inference.measure import Meter


def audit(originals, weights, destination, preview_dir=None):
    destination.mkdir(exist_ok=True, parents=True)
    if preview_dir:
        preview_dir.mkdir(exist_ok=True, parents=True)
    meter = Meter(Path("calibration/parameters.json"), weights, 36., 36., "cpu")
    outlines = json.loads(Path("dataset/reviewed-outlines.json").read_text())
    with Path("dataset/capture-index.csv").open() as f:
        records = list(csv.DictReader(f))
    rows = []
    # A stable image chosen by capture order before the test run.
    bench_source = min(r["source"] for r in records)
    for item in records:
        source = item["source"]
        image = read_image(originals/source)
        result = {"source": source, "slot": item["slot"], "split": item["split"]}
        points = np.float32(outlines[source]).reshape(-1,1,2)
        # Densify the traced edges so their image after undistortion can be curved.
        alpha = np.linspace(0, 1, 80, endpoint=False, dtype=np.float32)
        edge = (points[:,None,:,:]*(1-alpha[None,:,None,None]) +
                np.roll(points,-1,axis=0)[:,None,:,:]*alpha[None,:,None,None]).reshape(-1,1,2)
        matrix = np.asarray(meter.parameters["camera_matrix"])
        distortion = np.asarray(meter.parameters["distortion"])
        for mode in ("raw", "corrected"):
            raw = mode=="raw"
            prediction, overlay = meter.measure(image, raw=raw)
            for key in ("status","width_mm","height_mm","confidence"):
                result[f"{mode}_{key}"] = prediction[key]
            frame = image if raw else correct(image, meter.parameters)
            reference, reason = find_reference(frame)
            result[f"{mode}_reference"] = reason or "ok"
            for axis in ("width","height"):
                result[f"{mode}_traced_{axis}_mm"] = None
            if reason is None:
                contour = edge if raw else cv2.undistortPoints(edge, matrix, distortion, P=matrix)
                traced = metric_box(contour, reference,36.,36.)
                result[f"{mode}_traced_width_mm"], result[f"{mode}_traced_height_mm"] = traced
            if preview_dir and not raw and item["slot"] in ("envelope10_01.jpg", "envelope08_01.jpg"):
                cv2.imwrite(str(preview_dir/item["slot"]),overlay)
        rows.append(result)
    with (destination/"capture-results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats={}
    for split in ("train","val","test"):
        group=[r for r in rows if r["split"]==split]
        stats[split]={}
        for mode in ("raw","corrected"):
            accepted=[r for r in group if r[f"{mode}_status"]=="ok"]
            st={"accepted":len(accepted),"total":len(group)}
            for axis in ("width","height"):
                err=[abs(r[f"{mode}_{axis}_mm"]-r[f"{mode}_traced_{axis}_mm"]) for r in accepted]
                st[f"{axis}_disagreement_with_traced_outline_mm"]=(
                    dict(mean=mean(err),median=median(err),worst=max(err)) if err else None)
            stats[split][mode]=st
    stats["meaning"]="Both the model and traced outline use the same marker corners. Disagreement is a segmentation diagnostic. No hand-measured ground truth was supplied."
    # Warm model; include original HEIC decoding, orientation/resize, correction and prediction.
    path=originals/bench_source
    meter.measure(read_image(path))
    timings=[]
    for _ in range(5):
        t=perf_counter()
        pred,_=meter.measure(read_image(path))
        timings.append(perf_counter()-t)
    stats["five_warm_runs"]={"file":bench_source,"seconds":timings,"median_seconds":median(timings),
                             "min_seconds":min(timings),"max_seconds":max(timings),"last_status":pred["status"]}
    (destination/"capture-summary.json").write_text(json.dumps(stats,indent=2,allow_nan=False))
    # Functional negative controls on the benchmark photo, kept separate from physical photos.
    image=read_image(path)
    quad,reason=find_reference(image)
    if reason is None:
        ar=cv2.convexHull(quad.astype(np.float32))
        q=ar.mean(axis=0)+(ar-ar.mean(axis=0))*1.35
        removed=image.copy()
        cv2.fillConvexPoly(removed,q.astype(np.int32).reshape(-1,2),[195,195,195])
        blank=np.full_like(image,220)
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.fillConvexPoly(mask,q.astype(np.int32).reshape(-1,2),255)
        blank[mask>0]=image[mask>0]
        controls={key:meter.measure(frame)[0] for key,frame in
                  (("digitally_covered_marker",removed),("marker_on_empty_canvas",blank))}
        (destination/"negative-controls.json").write_text(json.dumps(controls,indent=2,allow_nan=False))
    print(json.dumps(stats,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("originals",type=Path)
    p.add_argument("--weights",type=Path,default=Path("models/runs/first/best.pth"))
    p.add_argument("--out",type=Path,default=Path("measurement/evidence"))
    p.add_argument("--preview-dir",type=Path)
    a=p.parse_args()
    audit(a.originals,a.weights,a.out,a.preview_dir)

