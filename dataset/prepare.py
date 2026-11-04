"""Undistort the fixed dataset and rasterize reviewed CVAT image polygons."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from calibration.camera import correct, read_image

BASE = Path("dataset")


def split_names(split):
    names = (BASE / "splits" / f"{split}.txt").read_text().splitlines()
    return [name.strip() for name in names if name.strip()]


def all_names():
    groups = [split_names(s) for s in ("train", "val", "test")]
    names = sum(groups, [])
    if len(set(names)) != len(names) or [len(g) for g in groups] != [49, 14, 7]:
        raise ValueError("Expected 49/14/7 distinct reserved filenames.")
    return names


def undistort_dataset(raw, calibration):
    params = json.loads(calibration.read_text())
    output = BASE / "images"
    output.mkdir(exist_ok=True)
    seen = {}
    for name in all_names():
        image = read_image(raw / name)
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest()
        if digest in seen:
            raise ValueError(f"Repeated source pixels: {name} and {seen[digest]}")
        seen[digest] = name
        if not cv2.imwrite(str(output / name), correct(image, params), [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise IOError(f"Failed to write {name}")
    print(f"Undistorted {len(seen)} original images. Label the envelope on these exported images.")


def import_masks(xml_path):
    records = ET.parse(xml_path).getroot().findall("image")
    images = {Path(node.attrib["name"]).name: node for node in records}
    names = all_names()
    if len(records) != len(names) or set(images) != set(names):
        raise ValueError("CVAT export must contain each reserved undistorted image exactly once.")
    output = BASE / "masks"
    output.mkdir(exist_ok=True)
    for name in names:
        node = images[name]
        polygons = [p for p in node.findall("polygon") if p.get("label") == "envelope"]
        if len(polygons) != 1:
            raise ValueError(f"{name}: expected one envelope polygon")
        coords = np.array([list(map(float, point.split(","))) for point in polygons[0].attrib["points"].split(";")])
        height, width = read_image(BASE / "images" / name).shape[:2]
        if (width, height) != (int(node.get("width")), int(node.get("height"))):
            raise ValueError(f"{name}: annotation size does not match the undistorted image")
        if len(coords) < 3 or not np.isfinite(coords).all() or (coords < 0).any() or (coords >= [width, height]).any():
            raise ValueError(f"{name}: invalid polygon")
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.rint(coords).astype(np.int32)], 255)
        if cv2.countNonZero(mask) < 16:
            raise ValueError(f"{name}: empty envelope annotation")
        if not cv2.imwrite(str(output / Path(name).with_suffix(".png")), mask):
            raise IOError(name)
    print(f"Imported {len(names)} binary masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    photos = commands.add_parser("undistort")
    photos.add_argument("--raw", type=Path, default=BASE / "raw")
    photos.add_argument("--calibration", type=Path, default=Path("calibration/parameters.json"))
    masks = commands.add_parser("import-cvat")
    masks.add_argument("xml", type=Path)
    args = parser.parse_args()
    if args.action == "undistort":
        undistort_dataset(args.raw, args.calibration)
    else:
        import_masks(args.xml)
