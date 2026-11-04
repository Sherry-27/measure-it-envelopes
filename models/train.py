"""Fine-tune on train only and freeze a validation-selected checkpoint."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from calibration.camera import read_image
from dataset.prepare import BASE, all_names, split_names
from models.network import build_model, tensor_image, working_frame


class Envelopes(Dataset):
    def __init__(self, split):
        all_names()
        self.names = split_names(split)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        original = read_image(BASE / "images" / name)
        mask = cv2.imread(str(BASE / "masks" / Path(name).with_suffix(".png")), 0)
        if mask is None or mask.shape != original.shape[:2]:
            raise ValueError(f"Missing or mismatched mask: {name}")
        frame = working_frame(original)
        mask = cv2.resize(mask, frame.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        y, x = np.nonzero(mask)
        if not x.size:
            raise ValueError(f"Empty mask: {name}")
        target = {"boxes": torch.tensor([[x.min(), y.min(), x.max()+1, y.max()+1]], dtype=torch.float32),
                  "labels": torch.ones(1, dtype=torch.int64),
                  "masks": torch.from_numpy(mask[None].astype(np.uint8)),
                  "image_id": torch.tensor(i), "area": torch.tensor([int(mask.sum())]),
                  "iscrowd": torch.zeros(1, dtype=torch.int64)}
        return tensor_image(frame), target


def intersection_over_union(pred, truth):
    return float((pred & truth).sum() / (pred | truth).sum().clamp(min=1))


@torch.inference_mode()
def validation(model, dataset, device, candidates):
    model.eval()
    rows = []
    for image, target in dataset:
        result = model([image.to(device)])[0]
        if len(result["scores"]):
            score = float(result["scores"][0])
            overlap = intersection_over_union(result["masks"][0, 0].cpu() >= 0.5, target["masks"][0].bool())
        else:
            score, overlap = 0., 0.
        runner_up = float(result["scores"][1]) if len(result["scores"]) > 1 else 0.
        rows.append((score, runner_up, overlap))
    summary = []
    for cut in candidates:
        # Reward retained correct detections while penalizing unreliable answers.
        tp = sum(score >= cut > second and iou >= 0.5 for score, second, iou in rows)
        fp = sum(score >= cut > second and iou < 0.5 for score, second, iou in rows)
        f1 = 2 * tp / max(len(rows) + tp + fp, 1)
        mean_iou = sum(iou for score, second, iou in rows if score >= cut > second) / len(rows)
        summary.append(dict(threshold=cut, f1=f1, mean_iou=mean_iou,
                            refused=sum(not (score >= cut > second) for score, second, _ in rows)))
    best = max(summary, key=lambda s: (s["f1"], s["mean_iou"], s["threshold"]))
    return best, summary


def train(config_path, out, device):
    cfg = json.loads(config_path.read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_ds, val_ds = Envelopes("train"), Envelopes("val")
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=lambda batch: tuple(zip(*batch)), num_workers=0)
    model = build_model(cfg, pretrained=True).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["lr"], momentum=0.9, weight_decay=0.0005)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "training.jsonl").exists() or (out / "best.pth").exists():
        raise ValueError("Choose a new run directory to preserve the existing evidence.")
    best_key = (-1., -1.)
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        losses = []
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        choice, grid = validation(model, val_ds, device, cfg["score_candidates"])
        key = (choice["mean_iou"], choice["f1"])
        if key > best_key:
            torch.save({"config": cfg, "state": model.state_dict(), "epoch": epoch,
                        "score_threshold": choice["threshold"]}, out / "best.tmp")
            (out / "best.tmp").replace(out / "best.pth")
            best_key = key
        line = json.dumps(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation=grid,
                               selected=choice, device=str(device)))
        print(line, flush=True)
        with (out / "training.jsonl").open("a") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("models/config.json"))
    p.add_argument("--out", type=Path, default=Path("models/runs/first"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args.config, args.out, args.device)
