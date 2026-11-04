"""Paired physical accuracy on ten or more independently ruler-measured instances."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from calibration.camera import read_image
from inference.measure import load_meter, options


def summarize(rows, mode):
    accepted = [r for r in rows if r[f"{mode}_status"] == "ok"]
    summary = {"total": len(rows), "accepted": len(accepted), "refused": len(rows)-len(accepted)}
    for axis in ("width", "height"):
        if not accepted:
            summary[axis] = None
            continue
        delta = [r[f"{mode}_{axis}_mm"]-r[f"hand_{axis}_mm"] for r in accepted]
        absolute = [abs(d) for d in delta]
        pct = [100*d/r[f"hand_{axis}_mm"] for d, r in zip(delta, accepted)]
        summary[axis] = dict(mae_mm=statistics.mean(absolute), median_abs_mm=statistics.median(absolute),
                             worst_abs_mm=max(absolute), mean_signed_pct=statistics.mean(pct),
                             mean_absolute_pct=statistics.mean(map(abs, pct)))
    return summary


def compare(args):
    with args.table.open(newline="", encoding="utf-8-sig") as source:
        truth = list(csv.DictReader(source))
    if len(truth) < 10 or len({r["id"] for r in truth}) != len(truth) or len({r["file"] for r in truth}) != len(truth):
        raise ValueError("Need at least ten distinct instances and separate photos.")
    for row in truth:
        for axis in ("width", "height"):
            key = f"hand_{axis}_mm"
            row[key] = float(row[key])
            if row[key] <= 0 or not math.isfinite(row[key]):
                raise ValueError(f"Invalid hand measurement: {row['id']}")
        if row["hand_width_mm"] < row["hand_height_mm"]:
            raise ValueError("Use the long edge as width in the hand table.")
    meter = load_meter(args)
    for row in truth:
        original = read_image(args.photos / row["file"])
        for mode in ("raw", "corrected"):
            pred, _ = meter.measure(original, raw=mode=="raw")
            for key in ("status", "confidence", "width_mm", "height_mm"):
                row[f"{mode}_{key}"] = pred[key]
            for axis in ("width", "height"):
                measured = pred[f"{axis}_mm"]
                hand = row[f"hand_{axis}_mm"]
                row[f"{mode}_{axis}_abs_mm"] = None if measured is None else abs(measured-hand)
                row[f"{mode}_{axis}_signed_pct"] = None if measured is None else 100*(measured-hand)/hand
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "instances.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    paired = [r for r in truth if all(r[f"{m}_status"] == "ok" for m in ("raw", "corrected"))]
    summary = {m: summarize(truth, m) for m in ("raw", "corrected")}
    summary["common_accepted"] = {m: summarize(paired, m) for m in ("raw", "corrected")}
    corrected = summary["corrected"]
    summary["mo1_met"] = corrected["accepted"] == len(truth) and all(corrected[a]["mae_mm"] <= 2 for a in ("width", "height"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True, type=Path)
    p.add_argument("--photos", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("measurement/evidence"))
    options(p)
    compare(p.parse_args())
