"""Final mask evaluation. Run after validation selection has finished."""
import argparse
import contextlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.network import load_model
from models.train import Envelopes, intersection_over_union


def encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))


@torch.inference_mode()
def evaluate(weights, output, device):
    model, threshold = load_model(weights, device)
    dataset = Envelopes("test")
    records, truth, preds, images = [], [], [], []
    tp = fp = 0
    for i, (image, target) in enumerate(dataset):
        height, width = image.shape[-2:]
        images.append(dict(id=i, width=width, height=height))
        gt_mask = target["masks"][0].bool()
        rle = encode(gt_mask)
        truth.append(dict(id=i+1, image_id=i, category_id=1, segmentation=rle,
                          area=float(mask_utils.area(rle)), bbox=mask_utils.toBbox(rle).tolist(), iscrowd=0))
        result = model([image.to(device)])[0]
        overlap, accepted = 0., False
        for j, (mask, score) in enumerate(zip(result["masks"], result["scores"])):
            mask = mask[0].cpu() >= 0.5
            score = float(score)
            preds.append(dict(image_id=i, category_id=1, segmentation=encode(mask), score=score))
            if j == 0 and score >= threshold and (len(result["scores"]) == 1 or float(result["scores"][1]) < threshold):
                accepted = True
                overlap = intersection_over_union(mask, gt_mask)
        tp += int(accepted and overlap >= 0.5)
        fp += int(accepted and overlap < 0.5)
        records.append(dict(file=dataset.names[i], mask_iou=overlap, detected=accepted))
    output.mkdir(parents=True, exist_ok=True)
    coco = COCO()
    coco.dataset = dict(info={}, images=images, annotations=truth, categories=[dict(id=1, name="envelope")])
    coco.createIndex()
    with (output / "coco-output.txt").open("w") as log, contextlib.redirect_stdout(log):
        if preds:
            metric = COCOeval(coco, coco.loadRes(preds), "segm")
            metric.evaluate()
            metric.accumulate()
            metric.summarize()
            ap, ap50 = (float(v) for v in metric.stats[:2])
        else:
            print("No predicted instances")
            ap = ap50 = 0.
    summary = dict(mask_iou=float(np.mean([r["mask_iou"] for r in records])),
                   mask_ap50=ap50, mask_ap50_95=ap, precision=tp/max(tp+fp, 1),
                   recall=tp/len(records), score_threshold=threshold, mask_threshold=0.5,
                   match_iou=0.5, working_long_edge_px=1024, test_images=records,
                   hardware=str(device), os=platform.platform(), torch=str(torch.__version__),
                   torchvision=str(torchvision.__version__))
    (output / "test-metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("models/evidence"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    evaluate(args.weights, args.out, args.device)
