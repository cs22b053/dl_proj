#!/usr/bin/env python3
import os
import argparse
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import YOLOv8UC  # Ensure this file defines your YOLOv8-UC model

# Dataset for YOLO format
class URPC2021Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=320):
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.lbl_dir = os.path.join(root_dir, 'labels', split)
        self.img_files = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png')))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # scales to [0,1]
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + '.txt')
        boxes, labels = [], []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, w, h = parts
                    cls = int(cls)
                    xc, yc, w, h = map(float, (xc, yc, w, h))
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        return image, {'boxes': boxes, 'labels': labels}

# Custom collate_fn to handle varying number of boxes per image
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets

# Inner-SIoU loss for bounding box regression
default_ratio = 1.2
class InnerSIoULoss(nn.Module):
    def __init__(self, ratio=default_ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, pred, gt):
        # pred, gt: [1,4]
        if gt.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        px1, py1, px2, py2 = pred[0]
        gx1, gy1, gx2, gy2 = gt[0]
        gw, gh = gx2 - gx1, gy2 - gy1
        cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        sw, sh = gw * self.ratio / 2, gh * self.ratio / 2
        sgx1, sgx2 = cx - sw, cx + sw
        sgy1, sgy2 = cy - sh, cy + sh
        ix1, iy1 = max(px1, sgx1), max(py1, sgy1)
        ix2, iy2 = min(px2, sgx2), min(py2, sgy2)
        iw = max(ix2 - ix1, torch.tensor(0.0, device=pred.device))
        ih = max(iy2 - iy1, torch.tensor(0.0, device=pred.device))
        inter = iw * ih
        pred_area = (px2 - px1) * (py2 - py1)
        sg_area = (sgx2 - sgx1) * (sgy2 - sgy1)
        union = pred_area + sg_area - inter + 1e-6
        iou_inner = inter / union
        return 1 - iou_inner

# Compute mean IoU on a DataLoader
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_iou = 0.0
    count = 0
    for imgs, targets in tqdm(loader, desc='Validation'):
        imgs = imgs.to(device)
        cls_preds, reg_preds = model(imgs)
        batch_size = imgs.size(0)
        for i in range(batch_size):
            boxes = targets[i]['boxes'].to(device)
            if boxes.numel() == 0:
                continue
            rp = reg_preds[i].mean(dim=(1,2))
            px, py, pw, ph = torch.sigmoid(rp[:4])
            x1 = px - pw/2; y1 = py - ph/2
            x2 = px + pw/2; y2 = py + ph/2
            gx1, gy1, gx2, gy2 = boxes[0]
            ix1, iy1 = max(x1, gx1), max(y1, gy1)
            ix2, iy2 = min(x2, gx2), min(y2, gy2)
            iw = max(ix2 - ix1, 0)
            ih = max(iy2 - iy1, 0)
            inter = iw * ih
            area_p = (x2 - x1) * (y2 - y1)
            area_g = (gx2 - gx1) * (gy2 - gy1)
            union = area_p + area_g - inter + 1e-6
            total_iou += inter / union
            count += 1
    return total_iou / max(count, 1)

# Training loop
def train(args):
    device = torch.device(args.device)
    train_ds = URPC2021Dataset(args.data_root, 'train', args.img_size)
    val_ds   = URPC2021Dataset(args.data_root, 'val',   args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)

    model = YOLOv8UC(num_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    ce_loss  = nn.CrossEntropyLoss()
    obj_loss = nn.BCEWithLogitsLoss()
    box_loss = InnerSIoULoss()

    best_iou = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            cls_preds, reg_preds = model(imgs)
            loss_batch = torch.tensor(0.0, device=device)
            for i in range(imgs.size(0)):
                boxes  = targets[i]['boxes'].to(device)
                labels = targets[i]['labels'].to(device)
                if boxes.numel() == 0:
                    continue
                cp = cls_preds[i].mean(dim=(1,2))
                rp = reg_preds[i].mean(dim=(1,2))
                px, py, pw, ph = torch.sigmoid(rp[:4])
                x1 = px - pw/2; y1 = py - ph/2
                x2 = px + pw/2; y2 = py + ph/2
                pred_box = torch.stack([x1,y1,x2,y2]).unsqueeze(0)
                gt_box   = boxes[0].unsqueeze(0)
                gt_lbl   = labels[0].unsqueeze(0)
                l_c = ce_loss(cp.unsqueeze(0), gt_lbl)
                l_o = obj_loss(rp[4].unsqueeze(0), torch.ones((1,), device=device))
                l_b = box_loss(pred_box, gt_box)
                loss_batch += (l_c + l_o + l_b)
            if loss_batch.item() > 0:
                loss_batch.backward()
                optimizer.step()
                epoch_loss += loss_batch.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        val_iou = validate(model, val_loader, device)
        print(f"Epoch {epoch} - Val mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model (mIoU={best_iou:.4f}) to {args.save_path}")

    print(f"Training complete. Best mIoU: {best_iou:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8-UC on URPC2021")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Prepared URPC2021 root (images/ & labels/)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='yolov8_uc_best.pth')
    args = parser.parse_args()
    try:
        train(args)
    except Exception as ex:
        print("❌ Error during training:", ex)
        traceback.print_exc()

