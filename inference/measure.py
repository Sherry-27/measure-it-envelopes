"""Measure one envelope in a new, original photograph."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from calibration.camera import correct, read_image
from inference.geometry import find_reference, metric_box
from models.network import load_model, tensor_image, working_frame


class Meter:
    def __init__(self, calibration, weights, marker_width, marker_height, device):
        self.parameters = json.loads(calibration.read_text())
        self.model, self.score_cut = load_model(weights, device)
        self.device = device
        if min(marker_width, marker_height) <= 0 or not np.isfinite([marker_width, marker_height]).all():
            raise ValueError("Supply the measured black marker dimensions in mm.")
        self.marker = marker_width, marker_height

    @torch.inference_mode()
    def measure(self, original, raw=False, refine_crop=False):
        target_w, target_h = self.parameters["image_size"]
        height, width = original.shape[:2]
        if [width, height] != [target_w, target_h]:
            if height * target_h == width * target_w:
                original = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
                height, width = original.shape[:2]
            if width * target_h != height * target_w:
                raise ValueError(
                    "Choose an uncropped original 4:3 camera photo (HEIC/JPG/PNG). "
                    "This file has a different aspect ratio, so its camera calibration cannot be reused. "
                    "Try Sample 1 first; do not select a screenshot or a cropped image.")
            original = cv2.resize(original, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frame = original.copy() if raw else correct(original, self.parameters)
        small = working_frame(frame)
        pred = self.model([tensor_image(small).to(self.device)])[0]
        score = float(pred["scores"][0]) if len(pred["scores"]) else None
        result = dict(status="object_not_found", width_mm=None, height_mm=None,
                      confidence=score, score_threshold=self.score_cut,
                      correction="raw" if raw else "undistorted")
        if score is None or score < self.score_cut:
            return result, frame
        if len(pred["scores"]) > 1 and float(pred["scores"][1]) >= self.score_cut:
            result["status"] = "multiple_objects"
            return result, frame
        binary = (pred["masks"][0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
        mask = cv2.resize(binary, frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        if refine_crop:
            from inference.crop import refine_mask
            result["coarse_confidence"] = score
            mask, fine_score, reason, bounds = refine_mask(frame, mask, self.model, self.device, self.score_cut)
            result.update(confidence=fine_score, segmentation_mode="refined_crop", crop_bounds_xyxy=bounds)
            if reason:
                result["status"] = reason
                return result, frame
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return result, frame
        contour = max(contours, key=cv2.contourArea)
        reference, reason = find_reference(frame)
        selected = mask.astype(bool)
        frame[selected] = (frame[selected] * 0.65 + np.array([60, 180, 60]) * 0.35).astype(np.uint8)
        cv2.drawContours(frame, [contour], -1, (60, 210, 60), 3)
        if reason:
            result["status"] = reason
            return result, frame
        cv2.polylines(frame, [np.rint(reference).astype(np.int32)], True, (0, 170, 255), 3)
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            result["status"] = "object_touches_edge"
            return result, frame
        try:
            width, height = metric_box(contour, reference, *self.marker)
        except ValueError:
            result["status"] = "unreliable_geometry"
            return result, frame
        result.update(status="ok", width_mm=width, height_mm=height)
        return result, frame


def load_meter(args):
    return Meter(args.calibration, args.weights, args.marker_width_mm, args.marker_height_mm, args.device)


def options(parser):
    parser.add_argument("--calibration", type=Path, default=Path("calibration/parameters.json"))
    parser.add_argument("--weights", type=Path, default=Path("models/runs/first/best.pth"))
    parser.add_argument("--marker-width-mm", default=36., type=float)
    parser.add_argument("--marker-height-mm", default=36., type=float)
    parser.add_argument("--device", default="cpu")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--out", type=Path, default=Path("overlay.jpg"))
    p.add_argument("--raw", action="store_true", help="Skip lens correction for a paired comparison on the same photograph.")
    p.add_argument("--refine-crop", action="store_true", help="Experimental second segmentation pass; restore the mask to the original camera coordinates.")
    options(p)
    args = p.parse_args()
    result, overlay = load_meter(args).measure(read_image(args.image), raw=args.raw, refine_crop=args.refine_crop)
    text = result["status"]
    if result["status"] == "ok":
        text = f'{result["width_mm"]:.1f} x {result["height_mm"]:.1f} mm  score {result["confidence"]:.2f}'
    scale = max(0.65, overlay.shape[1]/1400)
    cv2.putText(overlay, text, (12, int(35*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), max(3, int(5*scale)), cv2.LINE_AA)
    cv2.putText(overlay, text, (12, int(35*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), max(1, int(2*scale)), cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), overlay):
        raise IOError(args.out)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
