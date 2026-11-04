"""Rebuild the reviewed dataset from the original, named HEIC photos."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dataset.iphone import sensor_frame


def restore(originals, destination=Path("dataset"), calibration=Path("calibration/parameters.json")):
    with Path("dataset/capture-index.csv").open() as stream:
        index = list(csv.DictReader(stream))
    outlines = json.loads(Path("dataset/reviewed-outlines.json").read_text())
    params = json.loads(calibration.read_text())
    size = tuple(params["image_size"])
    matrix = np.asarray(params["camera_matrix"])
    distortion = np.asarray(params["distortion"])
    maps = cv2.initUndistortRectifyMap(matrix, distortion, None, matrix, size, cv2.CV_32FC1)
    for row in index:
        source = originals / row["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Original file differs from inventory: {source.name}")
    for name in ("images", "masks"):
        (destination / name).mkdir(exist_ok=True, parents=True)
    for row in index:
        source = originals / row["source"]
        frame = sensor_frame(source, long_edge=size[0])
        vertices = np.rint(outlines[source.name]).astype(np.int32)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        out_image = destination / "images" / row["slot"]
        out_mask = destination / "masks" / Path(row["slot"]).with_suffix(".png")
        if not cv2.imwrite(str(out_image), cv2.remap(frame, *maps, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise IOError(out_image)
        if not cv2.imwrite(str(out_mask), cv2.remap(mask, *maps, cv2.INTER_NEAREST)):
            raise IOError(out_mask)
    print(f"Restored {len(index)} calibrated image/mask pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("originals", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    restore(args.originals, args.destination)

