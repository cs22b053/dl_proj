#!/usr/bin/env python3
"""
eval.py

Evaluate a trained YOLOv8-UC model on the test split with COCO metrics.

Usage:
    python eval.py \
      --data_root /path/to/prepared \
      --model_path yolov8_uc_best.pth \
      --coco_json /path/to/instances_test.json \
      --batch_size 8 \
      --img_size 640 \
      --device cuda
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

# Dataset for evaluation\class URPCTestDataset(Dataset):
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
        h0, w0 = image.height, image.width
        img = self.transform(image)
        return img, info['id'], w0, h0

# decode single-image predictions using mean pooling (simplified)
def decode_preds(cls_pred, reg_pred, orig_w, orig_h):
    cp = cls_pred.mean(dim=(1,2))  # shape: (num_classes,)
    pobj = torch.sigmoid(reg_pred.mean(dim=(1,2))[4])
    score, cls_id = torch.max(cp, dim=0)
    score = score.item() * pobj.item()
    rp = reg_pred.mean(dim=(1,2))
    px, py, pw, ph = torch.sigmoid(rp[:4])
    # convert to absolute
    x1 = (px - pw/2) * orig_w
    y1 = (py - ph/2) * orig_h
    w = pw * orig_w
    h = ph * orig_h
    return cls_id.item(), score, [x1.item(), y1.item(), w.item(), h.item()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='Prepared data root')
    parser.add_argument('--model_path', required=True, help='Trained model checkpoint')
    parser.add_argument('--coco_json', required=True, help='COCO JSON for test set')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results', default='results.json', help='Output JSON file')
    args = parser.parse_args()

    device = torch.device(args.device)
    # Load model
    model = YOLOv8UC(num_classes=5).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # DataLoader
    dataset = URPCTestDataset(args.data_root, args.coco_json, args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    results = []
    with torch.no_grad():
        for imgs, img_ids, w0s, h0s in loader:
            imgs = imgs.to(device)
            cls_preds, reg_preds = model(imgs)
            for i in range(imgs.size(0)):
                img_id = img_ids[i].item()
                w0, h0 = w0s[i].item(), h0s[i].item()
                cls_p = cls_preds[i]; reg_p = reg_preds[i]
                cls_id, score, bbox = decode_preds(cls_p, reg_p, w0, h0)
                results.append({
                    'image_id': img_id,
                    'category_id': int(cls_id),
                    'bbox': bbox,
                    'score': score,
                })

    # Save results JSON
    with open(args.results, 'w') as f:
        json.dump(results, f)
    print(f"Saved {len(results)} detections to {args.results}")

    # COCO evaluation
    coco_gt = COCO(args.coco_json)
    coco_dt = coco_gt.loadRes(args.results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = dataset.img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

