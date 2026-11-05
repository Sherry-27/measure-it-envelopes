"""Decode iPhone HEIC files into a consistent sensor coordinate system."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()


def sensor_frame(path, long_edge=1200):
    with Image.open(path) as source:
        orientation = source.info.get("original_orientation", 1) or 1
        rgb = np.array(source.convert("RGB"))
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if orientation == 6:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 8:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 3:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif orientation != 1:
        raise ValueError(f"Unsupported orientation {orientation}")
    if frame.shape[:2] != (3024, 4032):
        raise ValueError(f"Expected uncropped iPhone 11 Pro sensor pixels, got {frame.shape[1::-1]}")
    return cv2.resize(frame, (long_edge, round(long_edge * 3/4)), interpolation=cv2.INTER_AREA)
