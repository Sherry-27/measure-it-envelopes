"""Camera calibration for a single lens and fixed sensor resolution."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    frame = cv2.imread(str(path), flags)
    if frame is None:
        raise ValueError(f"Unreadable image: {path}. Export the original as JPEG or PNG.")
    return frame


def correct(frame, parameters):
    size = list(frame.shape[1::-1])
    if size != parameters["image_size"]:
        raise ValueError(f"Expected calibrated resolution {parameters['image_size']}, got {size}")
    matrix = np.asarray(parameters["camera_matrix"], dtype=float)
    distortion = np.asarray(parameters["distortion"], dtype=float)
    return cv2.undistort(frame, matrix, distortion, None, matrix)


def find_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1400 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, corners = cv2.findChessboardCornersSB(small, (7, 9))
    if not found:
        return None
    corners[:, :, 0] *= gray.shape[1] / small.shape[1]
    corners[:, :, 1] *= gray.shape[0] / small.shape[0]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)


def calibrate(source, square, output):
    if square <= 0 or not np.isfinite(square):
        raise ValueError("Supply the measured square size in mm.")
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    grid = np.zeros((63, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2) * square
    accepted, rejected, points = [], [], []
    size = None
    for path in paths:
        frame = read_image(path)
        frame_size = frame.shape[1::-1]
        if size is None:
            size = frame_size
        if size != frame_size:
            raise ValueError(f"Resolution changed in {path.name}. Check orientation and export settings.")
        corners = find_board(frame)
        if corners is None:
            rejected.append(path.name)
            continue
        accepted.append(path.name)
        points.append(corners)
    if len(points) < 20:
        raise ValueError(f"Only {len(points)} usable board photos; need 20. Rejected: {rejected}")
    rms, matrix, dist, rotations, translations = cv2.calibrateCamera(
        [grid] * len(points), points, size, None, None
    )
    per_view = {}
    for name, observed, rot, trans in zip(accepted, points, rotations, translations):
        projected = cv2.projectPoints(grid, rot, trans, matrix, dist)[0]
        per_view[name] = float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=2))))
    parameters = dict(image_size=list(size), square_mm=square, camera_matrix=matrix.tolist(),
                      distortion=dist.ravel().tolist(), rms_px=float(rms), accepted=accepted,
                      rejected=rejected, per_view_rms_px=per_view)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(parameters, indent=2, allow_nan=False))
    before = read_image(source / accepted[0])
    after = correct(before, parameters)
    pair = cv2.hconcat([before, after])
    preview = cv2.resize(pair, (1400, round(1400 * pair.shape[0] / pair.shape[1])))
    cv2.imwrite(str(output.with_suffix(".preview.jpg")), preview)
    print(json.dumps({"accepted": len(accepted), "rejected": rejected, "rms_px": rms}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--square-mm", required=True, type=float)
    parser.add_argument("--output", type=Path, default=Path("calibration/parameters.json"))
    args = parser.parse_args()
    calibrate(args.source, args.square_mm, args.output)
