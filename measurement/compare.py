"""Compare ruler readings with paired raw and corrected measurements.

A specimen_id identifies the physical envelope, not the photograph. Repeated
captures remain separate trials and do not count as additional specimens.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


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


def load_truth(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required = {"id", "file", "specimen_id", "hand_width_mm", "hand_height_mm", "ruler_division_mm"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Use the hand-readings template, including specimen identity and ruler division.")
    if len(rows) < 10 or len({r["id"] for r in rows}) != len(rows) or len({r["file"] for r in rows}) != len(rows):
        raise ValueError("Need at least ten distinct photo trials with unique IDs and filenames.")
    for row in rows:
        if not all(row[k].strip() for k in required):
            raise ValueError(f"Missing physical reading or specimen identity for {row['file']}.")
        for key in ("hand_width_mm", "hand_height_mm", "ruler_division_mm"):
            try:
                row[key] = float(row[key])
            except ValueError:
                raise ValueError(f"Enter a measured number for {row['file']}: {key}") from None
            if row[key] <= 0 or not math.isfinite(row[key]):
                raise ValueError(f"Invalid hand measurement: {row['id']}")
        if row["hand_width_mm"] < row["hand_height_mm"]:
            raise ValueError("Use the long edge as width in the hand table.")
    return rows


def read_predictions(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        records = list(csv.DictReader(source))
    lookup = {r["source"]: r for r in records}
    if len(lookup) != len(records):
        raise ValueError("Prediction table contains duplicate source filenames.")
    return lookup


def saved_prediction(record, mode):
    result = {}
    for key in ("status", "confidence", "width_mm", "height_mm"):
        value = record[f"{mode}_{key}"]
        if key != "status":
            value = None if value == "" else float(value)
            if value is not None and not math.isfinite(value):
                raise ValueError("Prediction values must be finite or empty for a refusal.")
        result[key] = value
    if result["status"] == "ok" and any(result[k] is None or result[k] <= 0 for k in ("width_mm", "height_mm")):
        raise ValueError("An accepted prediction needs two positive dimensions.")
    if result["status"] != "ok" and any(result[k] is not None for k in ("width_mm", "height_mm")):
        raise ValueError("A refused prediction must not contain dimensions.")
    return result


def compare(args):
    truth = load_truth(args.table)
    cached = read_predictions(args.predictions) if args.predictions else None
    if cached is not None:
        missing = [r["file"] for r in truth if r["file"] not in cached]
        if missing:
            raise ValueError(f"No saved prediction for {missing}")
    else:
        from calibration.camera import read_image
        from inference.measure import load_meter
        meter = load_meter(args)
    for row in truth:
        original = read_image(args.photos / row["file"]) if cached is None else None
        row["split"] = cached[row["file"]].get("split", "unknown") if cached is not None else "new_capture"
        for mode in ("raw", "corrected"):
            if cached is None:
                pred, _ = meter.measure(original, raw=mode == "raw")
            else:
                pred = saved_prediction(cached[row["file"]], mode)
            for key in ("status", "confidence", "width_mm", "height_mm"):
                row[f"{mode}_{key}"] = pred[key]
            for axis in ("width", "height"):
                measured = pred[f"{axis}_mm"]
                hand = row[f"hand_{axis}_mm"]
                row[f"{mode}_{axis}_abs_mm"] = None if measured is None else abs(measured-hand)
                row[f"{mode}_{axis}_signed_pct"] = None if measured is None else 100*(measured-hand)/hand
                row[f"{mode}_{axis}_abs_pct"] = None if measured is None else 100*abs(measured-hand)/hand
    paired = [r for r in truth if all(r[f"{m}_status"] == "ok" for m in ("raw", "corrected"))]
    summary = {m: summarize(truth, m) for m in ("raw", "corrected")}
    summary["common_accepted"] = {m: summarize(paired, m) for m in ("raw", "corrected")}
    summary["trials_per_specimen"] = dict(Counter(r["specimen_id"] for r in truth))
    summary["unique_specimens"] = len(summary["trials_per_specimen"])
    summary["by_split"] = {
        split: {m: summarize([r for r in truth if r["split"] == split], m) for m in ("raw", "corrected")}
        for split in sorted({r["split"] for r in truth})
    }
    summary["paired_mae_change_mm"] = {
        axis: (summary["common_accepted"]["corrected"][axis]["mae_mm"] -
               summary["common_accepted"]["raw"][axis]["mae_mm"]) if paired else None
        for axis in ("width", "height")
    }
    summary["paired_mae_change_meaning"] = "Corrected minus raw on the same accepted photos; negative is improvement."
    corrected = summary["corrected"]
    scope_met = cached is None and summary["unique_specimens"] >= 10
    summary["mo1_scope_met"] = scope_met
    summary["mo1_met"] = scope_met and corrected["accepted"] == len(truth) and all(
        corrected[a]["mae_mm"] <= 2 for a in ("width", "height"))
    summary["scope_note"] = (
        "MO-1 originally specified fresh photos of ten physical envelopes. Existing dataset captures "
        "and repeated views can support a descriptive ruler comparison, but do not establish that scope. "
        "Split summaries retain any training-data reuse; photo trials are not independent specimens.")
    summary["inputs"] = {"hand_readings_sha256": hashlib.sha256(args.table.read_bytes()).hexdigest()}
    if args.predictions:
        summary["inputs"]["predictions_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "instances.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, allow_nan=False))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True, type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--photos", type=Path, help="Folder of new original camera photos.")
    source.add_argument("--predictions", type=Path, help="Existing audit_outputs CSV; skips model inference.")
    p.add_argument("--out", type=Path, default=Path("measurement/evidence/physical"))
    p.add_argument("--calibration", type=Path, default=Path("calibration/parameters.json"))
    p.add_argument("--weights", type=Path, default=Path("models/runs/first/best.pth"))
    p.add_argument("--marker-width-mm", default=36., type=float)
    p.add_argument("--marker-height-mm", default=36., type=float)
    p.add_argument("--device", default="cpu")
    compare(p.parse_args())
