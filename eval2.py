#!/usr/bin/env python3
"""
eval.py

Evaluate a trained YOLOv8-UC model on the test split using COCO metrics.
"""
import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import YOLOv8UC  # Your model definition

# Dataset for evaluation
class URPCTestDataset(Dataset):
    def __init__(self, data_root, coco_json, img_size):
        self.data_root = data_root
        self.coco = COCO(coco_json)
        self.img_ids = self.coco.getImgIds()
        self.img_infos = self.coco.loadImgs(self.img_ids)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.img_size = img_size

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        info = self.img_infos[idx]
        img_path = os.path.join(self.data_root, 'images', 'test', info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.width, image.height
        img = self.transform(image)
        return img, info['id'], orig_w, orig_h

# Decode predictions for a single image
def decode_preds(cls_pred, reg_pred, orig_w, orig_h):
    # cls_pred: (C, H, W), reg_pred: (5, H, W)
    cp = cls_pred.mean(dim=(1,2))  # (num_classes,)
    score_cls, cls_id = cp.max(dim=0)
    pobj = torch.sigmoid(reg_pred[4].mean())
    score = score_cls.item() * pobj.item()

    rp = reg_pred.mean(dim=(1,2))
    px, py, pw, ph = torch.sigmoid(rp[:4])
    # convert to absolute [x1, y1, w, h]
    x_center = px.item() * orig_w
    y_center = py.item() * orig_h
    w = pw.item() * orig_w
    h = ph.item() * orig_h
    x1 = x_center - w/2
    y1 = y_center - h/2
    return int(cls_id.item()), float(score), [x1, y1, w, h]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8-UC on URPC2021 test set')
    parser.add_argument('--data_root', required=True, help='Prepared data root with images/test')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--coco_json', required=True, help='COCO JSON for test annotations')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results', default='detections.json', help='Output JSON file')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = YOLOv8UC(num_classes=5).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # DataLoader
    dataset = URPCTestDataset(args.data_root, args.coco_json, args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    predictions = []
    with torch.no_grad():
        for imgs, img_ids, orig_ws, orig_hs in loader:
            imgs = imgs.to(device)
            cls_preds, reg_preds = model(imgs)
            for i in range(imgs.size(0)):
                cls_id, score, bbox = decode_preds(
                    cls_preds[i], reg_preds[i], orig_ws[i], orig_hs[i]
                )
                predictions.append({
                    'image_id': int(img_ids[i]),
                    'category_id': cls_id,
                    'bbox': bbox,
                    'score': score,
                })

    # Save to JSON
    with open(args.results, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} detections to {args.results}")

    # COCO evaluation
    coco_gt = COCO(args.coco_json)
    coco_dt = coco_gt.loadRes(args.results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = dataset.img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

