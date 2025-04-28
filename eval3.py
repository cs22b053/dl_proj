#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from train import URPC2021Dataset   # make sure train.py is in PYTHONPATH
from model import YOLOv8UC

# torchmetrics for detection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate_model(
    data_root="/scratch/cs22b053/dl/dl_proj/URPC2021_prepared",
    checkpoint="yolov8_uc_best.pth",
    batch_size=8,
    img_size=640,
    device="cuda"
):
    # 1) DataLoader
    test_ds = URPC2021Dataset(data_root, split="test", img_size=img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            [x[1] for x in b]
        ),
    )

    # 2) Model
    model = YOLOv8UC(num_classes=5)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 3) Metric
    metric = MeanAveragePrecision(class_metrics=True).to(device)

    # 4) Inference + collect
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            cls_out, reg_out = model(imgs)

        for i in range(imgs.size(0)):
            t = targets[i]
            gt_boxes = t["boxes"].to(device)
            gt_labels = t["labels"].to(device)

            # if no GT, skip
            if gt_boxes.numel() == 0:
                continue

            # decode prediction (mean pooling)
            cp = cls_out[i].mean(dim=(1,2)).softmax(-1)            # (C,)
            rp = reg_out[i].mean(dim=(1,2))                       # (5,)
            score_obj = torch.sigmoid(rp[4])
            score_cls, cls_id = cp.max(dim=0)
            score = (score_cls * score_obj).unsqueeze(0)

            px, py, pw, ph = torch.sigmoid(rp[:4])
            # scale to pixel coords (img_size×img_size)
            x1 = (px - pw/2) * img_size
            y1 = (py - ph/2) * img_size
            x2 = (px + pw/2) * img_size
            y2 = (py + ph/2) * img_size
            pred_boxes = torch.tensor([[x1, y1, x2, y2]], device=device)

            metric.update(
                [
                    {
                        "boxes": pred_boxes,
                        "scores": score,
                        "labels": cls_id.unsqueeze(0),
                    }
                ],
                [
                    {
                        "boxes": gt_boxes,
                        "labels": gt_labels,
                    }
                ],
            )

    # 5) Compute & print
    results = metric.compute()
    print(f"mAP@0.5:0.95  = {results['map']:.4f}")
    print(f"mAP@0.5       = {results['map_50']:.4f}")
    print(f"mAP@0.75      = {results['map_75']:.4f}")
    print(f"Recall@100    = {results['recall']:.4f}")

    # 6) Model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

if __name__ == "__main__":
    evaluate_model()

