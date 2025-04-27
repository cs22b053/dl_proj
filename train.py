import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import YOLOv8UC

class URPC2021Dataset(Dataset):
    """Dataset for URPC2021 in YOLO format."""
    def __init__(self, root_dir, split='train', img_size=640):
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.lbl_dir = os.path.join(root_dir, 'labels', split)
        self.img_files = sorted(os.listdir(self.img_dir))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        # Load image (placeholder code; replace with PIL or cv2 in practice)
        image = transforms.functional.pil_to_tensor(
            transforms.functional.resize(
                transforms.functional.pil_from_bytes(open(img_path, 'rb').read()),
                (self.img_size, self.img_size)
            )
        )
        image = image.float() / 255.0  # normalize to [0,1]
        # Load labels (YOLO txt)
        boxes = []
        labels = []
        lbl_path = os.path.join(self.lbl_dir, self.img_files[idx].replace('.jpg', '.txt'))
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = parts
                        cls = int(cls)
                        x, y, w, h = map(float, (x, y, w, h))
                        # Convert YOLO (x_center,y_center,w,h) to (x1,y1,x2,y2)
                        xmin = x - w/2; xmax = x + w/2
                        ymin = y - h/2; ymax = y + h/2
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls)
        boxes = torch.tensor(boxes) if boxes else torch.zeros((0,4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        return image, {'boxes': boxes, 'labels': labels}

class InnerSIoULoss(nn.Module):
    """Inner-SIoU loss: 1 - IoU(pred_box, scaled_gt_box)."""
    def __init__(self, ratio=1.2):
        super(InnerSIoULoss, self).__init__()
        self.ratio = ratio
    def forward(self, pred, gt):
        # pred, gt: tensors of shape (1,4) = [x1,y1,x2,y2]
        if gt.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        # Compute intersection over union with scaled GT
        px1, py1, px2, py2 = pred[0]
        gx1, gy1, gx2, gy2 = gt[0]
        gw = gx2 - gx1; gh = gy2 - gy1
        cx = (gx1 + gx2) / 2; cy = (gy1 + gy2) / 2
        # Scale ground truth box
        sw = gw * self.ratio / 2; sh = gh * self.ratio / 2
        sgx1 = cx - sw; sgx2 = cx + sw
        sgy1 = cy - sh; sgy2 = cy + sh
        # Intersection
        ix1 = max(px1, sgx1); iy1 = max(py1, sgy1)
        ix2 = min(px2, sgx2); iy2 = min(py2, sgy2)
        iw = max(ix2 - ix1, torch.tensor(0.0, device=px1.device))
        ih = max(iy2 - iy1, torch.tensor(0.0, device=py1.device))
        inter = iw * ih
        # Union
        pred_area = (px2 - px1) * (py2 - py1)
        sg_area = (sgx2 - sgx1) * (sgy2 - sgy1)
        union = pred_area + sg_area - inter + 1e-6
        iou_inner = inter / union
        return 1 - iou_inner

def train_model(root_dir, epochs=50, batch_size=8, lr=1e-3, device='cuda'):
    # Create datasets and loaders
    train_dataset = URPC2021Dataset(root_dir, split='train', img_size=640)
    val_dataset   = URPC2021Dataset(root_dir, split='val',   img_size=640)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = YOLOv8UC(num_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Losses
    ce_loss = nn.CrossEntropyLoss()         # for classification
    obj_loss = nn.BCEWithLogitsLoss()       # for objectness
    box_loss = InnerSIoULoss(ratio=1.2)     # Inner-SIoU

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            cls_preds, reg_preds = model(imgs)
            # Simplified loss: assume 1 object per image
            loss_batch = 0.0
            for i in range(imgs.size(0)):
                boxes = targets['boxes'][i]
                labels = targets['labels'][i]
                if boxes.numel() == 0:
                    continue
                # Aggregate predictions (mean over spatial dims) as a simple proxy
                cls_pred = cls_preds[i].mean(dim=[1,2])  # shape (num_classes,)
                reg_pred = reg_preds[i].mean(dim=[1,2])  # shape (5,)
                # Decode predictions (sigmoid for coords, objectness)
                px = torch.sigmoid(reg_pred[0])
                py = torch.sigmoid(reg_pred[1])
                pw = torch.sigmoid(reg_pred[2])
                ph = torch.sigmoid(reg_pred[3])
                pobj = torch.sigmoid(reg_pred[4])
                pred_box = torch.tensor([px, py, pw, ph], device=device)
                # Convert to (x1,y1,x2,y2) in normalized coords
                x1 = px - pw/2; y1 = py - ph/2
                x2 = px + pw/2; y2 = py + ph/2
                pred_xyxy = torch.stack([x1, y1, x2, y2]).unsqueeze(0)
                # Ground truth (take the first object as example)
                gt_box = boxes[0].to(device)  # [x1,y1,x2,y2] normalized
                gt_label = labels[0].to(device)
                # Compute losses
                loss_cls = ce_loss(cls_pred.unsqueeze(0), gt_label.unsqueeze(0))
                loss_obj = obj_loss(reg_pred[4].unsqueeze(0), torch.tensor([1.0], device=device))
                loss_box = box_loss(pred_xyxy, gt_box.unsqueeze(0))
                loss_batch += loss_cls + loss_obj + loss_box
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            running_loss += loss_batch.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")
        # (Optional) validation loop can be added here
    # Save trained model
    torch.save(model.state_dict(), "yolov8_uc.pth")

if __name__ == "__main__":
    data_root = "/path/to/URPC2021"  # replace with actual path
    train_model(data_root)
