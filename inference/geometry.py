"""Map an envelope contour into the coordinate system of marker 0."""
import cv2
import numpy as np


def find_reference(frame):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(frame)
    matches = [] if ids is None else [c[0] for c, key in zip(corners, ids.ravel()) if key == 0]
    if not matches:
        return None, "reference_not_found"
    if len(matches) > 1:
        return None, "ambiguous_reference"
    quad = matches[0]
    sides = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
    if sides.min() < 32:
        return None, "reference_too_small"
    return quad, None


def metric_box(contour, reference, width_mm, height_mm):
    if not np.isfinite([width_mm, height_mm]).all() or min(width_mm, height_mm) <= 0:
        raise ValueError("Marker dimensions must be positive measured millimetres.")
    plane = np.float32([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]])
    to_mm = cv2.getPerspectiveTransform(reference.astype(np.float32), plane)
    homogeneous = cv2.convertPointsToHomogeneous(contour.astype(np.float32)).reshape(-1, 3)
    denominators = homogeneous @ to_mm[2]
    if np.any(np.abs(denominators) < 1e-6) or denominators.min()*denominators.max() <= 0:
        raise ValueError("Contour crosses the plane horizon")
    metric = cv2.perspectiveTransform(contour.astype(np.float32), to_mm)
    rect = cv2.minAreaRect(metric)
    long_side, short_side = sorted(rect[1], reverse=True)
    if not np.isfinite(rect[1]).all() or short_side <= 0:
        raise ValueError("Degenerate outline")
    return float(long_side), float(short_side)

