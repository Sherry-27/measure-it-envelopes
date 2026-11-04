"""Inspect the printed grid in each supplied photo in its own pixel coordinates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.camera import find_board, read_image


def inspect_views(source, output):
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise ValueError("No input images")
    output.mkdir(parents=True, exist_ok=True)
    grid = np.float32([(x, y) for y in range(9) for x in range(7)])
    parity = (grid.sum(axis=1).astype(int) % 2).astype(bool)
    rows, detections, tiles = [], {}, []
    for path in paths:
        frame = read_image(path)
        height, width = frame.shape[:2]
        corners = find_board(frame)
        row = dict(file=path.name, width_px=width, height_px=height,
                   corners=0 if corners is None else len(corners),
                   planar_check_rms_px=None, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if corners is not None:
            points = corners.reshape(-1, 2)
            detections[path.name] = points.tolist()
            # Each point is scored against a planar map fitted without that point.
            errors = []
            for fit in (parity, ~parity):
                mapping, _ = cv2.findHomography(grid[fit], points[fit], 0)
                predicted = cv2.perspectiveTransform(grid[~fit, None, :], mapping)[:, 0, :]
                errors.extend(np.linalg.norm(predicted - points[~fit], axis=1))
            row["planar_check_rms_px"] = round(float(np.sqrt(np.mean(np.square(errors)))), 4)
            cv2.drawChessboardCorners(frame, (7, 9), corners, True)
        rows.append(row)
        ratio = min(280 / width, 180 / height)
        preview = cv2.resize(frame, (round(width*ratio), round(height*ratio)), interpolation=cv2.INTER_AREA)
        tile = np.full((214, 296, 3), 245, np.uint8)
        h, w = preview.shape[:2]
        x = (296-w)//2
        tile[5:5+h, x:x+w] = preview
        cv2.putText(tile, f"{path.stem}  |  {row['corners']}/63", (9, 203),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(tile)
    with (output / "views.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "corners.json").write_text(json.dumps(detections, separators=(",", ":")))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 245))
    sheet = cv2.vconcat([cv2.hconcat(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
    if not cv2.imwrite(str(output / "detected-views.jpg"), sheet):
        raise IOError("Could not write board contact sheet")
    values = [r["planar_check_rms_px"] for r in rows if r["planar_check_rms_px"] is not None]
    print(json.dumps(dict(received=len(rows), full_grids=len(values),
                          image_sizes=len({(r["width_px"], r["height_px"]) for r in rows}),
                          planar_rms_mean_px=float(np.mean(values)) if values else None,
                          planar_rms_median_px=float(np.median(values)) if values else None,
                          planar_rms_worst_px=max(values, default=None)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inspect_views(args.source, args.output)
