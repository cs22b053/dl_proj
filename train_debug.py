# train_debug.py
import os
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import YOLOv8UC  # Your YOLOv8-UC model definition

# ======== DATASET ========
class URPC2021Dataset(Dataset):
    """Dataset for URPC2021 in YOLO format."""
    def __init__(self, root_dir, split='train', img_size=320):
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.lbl_dir = os.path.join(root_dir, 'labels', split)
        self.img_files = sorted(os.listdir(self.img_dir))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Load labels
        boxes, labels = [], []
        lbl_path = os.path.join(self.lbl_dir, img_name.rsplit('.',1)[0] + '.txt')
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    cls, xc, yc, w, h = line.split()
                    cls = int(cls)
                    xc, yc, w, h = map(float, (xc, yc, w, h))
                    x1 = xc - w/2; y1 = yc - h/2
                    x2 = xc + w/2; y2 = yc + h/2
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
        boxes  = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4))
        labels = torch.tensor(labels, dtype=torch.long)     if labels else torch.zeros((0,), dtype=torch.long)
        return image, {'boxes': boxes, 'labels': labels}

# ======== INNER-SIoU LOSS ========
class InnerSIoULoss(nn.Module):
    """Inner-SIoU loss: 1 - IoU(pred_box, scaled_gt_box)."""
    def __init__(self, ratio=1.2):
        super().__init__()
        self.ratio = ratio
    def forward(self, pred, gt):
        # pred, gt: [1,4] = x1,y1,x2,y2 (normalized)
        if gt.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        px1, py1, px2, py2 = pred[0]
        gx1, gy1, gx2, gy2 = gt[0]
        gw, gh = gx2-gx1, gy2-gy1
        cx, cy = (gx1+gx2)/2, (gy1+gy2)/2
        sw, sh = gw*self.ratio/2, gh*self.ratio/2
        sgx1, sgx2 = cx-sw, cx+sw
        sgy1, sgy2 = cy-sh, cy+sh
        ix1, iy1 = max(px1, sgx1), max(py1, sgy1)
        ix2, iy2 = min(px2, sgx2), min(py2, sgy2)
        iw = max(ix2-ix1, torch.tensor(0.0, device=px1.device))
        ih = max(iy2-iy1, torch.tensor(0.0, device=py1.device))
        inter = iw * ih
        pred_area = (px2-px1)*(py2-py1)
        sg_area   = (sgx2-sgx1)*(sgy2-sgy1)
        union = pred_area + sg_area - inter + 1e-6
        iou_inner = inter / union
        return 1 - iou_inner

# ======== TRAINING LOOP ========
def train_model(root_dir,
                epochs=10,
                batch_size=1,
                img_size=320,
                lr=1e-3,
                device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INIT] Device={device}, epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
    # Data
    train_ds = URPC2021Dataset(root_dir, 'train', img_size)
    val_ds   = URPC2021Dataset(root_dir, 'val',   img_size)
    print(f"[DATA] train={len(train_ds)} images, val={len(val_ds)} images")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    # Model & optimizer
    model     = YOLOv8UC(num_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Losses
    ce_loss  = nn.CrossEntropyLoss()
    obj_loss = nn.BCEWithLogitsLoss()
    box_loss = InnerSIoULoss(ratio=1.2)
    print("[START] Entering training loop")
    for epoch in range(1, epochs+1):
        print(f" Epoch {epoch}/{epochs}")
        model.train()
        epoch_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            cls_preds, reg_preds = model(imgs)
            # initialize as tensor so .backward() works
            batch_loss = torch.tensor(0.0, device=device)
            # assume ≤1 object per image
            for i in range(imgs.size(0)):
                boxes  = targets['boxes'][i]
                labels = targets['labels'][i]
                if boxes.numel() == 0:
                    continue
                cls_pred = cls_preds[i].mean(dim=(1,2))  # (C,)
                reg_pred = reg_preds[i].mean(dim=(1,2))  # (5,)
                # decode
                px = torch.sigmoid(reg_pred[0])
                py = torch.sigmoid(reg_pred[1])
                pw = torch.sigmoid(reg_pred[2])
                ph = torch.sigmoid(reg_pred[3])
                pobj = torch.sigmoid(reg_pred[4])
                x1, y1 = px-pw/2, py-ph/2
                x2, y2 = px+pw/2, py+ph/2
                pred_box = torch.stack([x1,y1,x2,y2]).unsqueeze(0)
                gt_box   = boxes[0].to(device).unsqueeze(0)
                gt_label = labels[0].to(device).unsqueeze(0)
                # compute losses
                loss_c = ce_loss(cls_pred.unsqueeze(0), gt_label)
                loss_o = obj_loss(reg_pred[4].unsqueeze(0), torch.ones((1,),device=device))
                loss_b = box_loss(pred_box, gt_box)
                batch_loss = batch_loss + loss_c + loss_o + loss_b
            # skip if no objects (batch_loss is still zero)
            if batch_loss.item() == 0.0:
                continue
            # backward + step with OOM catch
            try:
                batch_loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(" ⚠️ CUDA OOM, emptying cache and skipping this batch")
                    torch.cuda.empty_cache()
                else:
                    raise
            epoch_loss += batch_loss.item()
        print(f"  --> Epoch {epoch} total loss: {epoch_loss:.4f}")
        # GPU memory print
        if device.type == 'cuda':
            os.system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    # save
    torch.save(model.state_dict(), "yolov8_uc_debug.pth")
    print("[DONE] Model saved to yolov8_uc_debug.pth")

if __name__ == "__main__":
    print("==>> Running train_debug.py")
    try:
        train_model(root_dir="/workspaces/dl_proj/URPC2021",
                    epochs=10,
                    batch_size=1,
                    img_size=320,
                    lr=1e-3)
    except Exception as ex:
        print("❌ Uncaught exception:", ex)
        traceback.print_exc()
    print("==>> train_debug.py has exited")
