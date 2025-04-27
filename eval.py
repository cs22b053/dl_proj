import torch
from torch.utils.data import DataLoader
from train import URPC2021Dataset
from model import YOLOv8UC

# Use torchmetrics for detection metrics
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None

def evaluate_model(data_root, batch_size=8, device='cuda'):
    test_dataset = URPC2021Dataset(data_root, split='test', img_size=640)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = YOLOv8UC(num_classes=5)
    model.load_state_dict(torch.load("yolov8_uc.pth", map_location=device))
    model.to(device).eval()
    
    all_preds = []
    all_targets = []
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            cls_out, reg_out = model(imgs)
        for i in range(imgs.size(0)):
            boxes = targets['boxes'][i]
            labels = targets['labels'][i]
            # Skip if no object
            if boxes.numel() == 0:
                continue
            # Simplified: take average predictions (as in train)
            cls_pred = cls_out[i].mean(dim=[1,2]).softmax(dim=-1)  # (num_classes,)
            reg_pred = reg_out[i].mean(dim=[1,2])  # (5,)
            px = torch.sigmoid(reg_pred[0])
            py = torch.sigmoid(reg_pred[1])
            pw = torch.sigmoid(reg_pred[2])
            ph = torch.sigmoid(reg_pred[3])
            pobj = torch.sigmoid(reg_pred[4]).item()
            pred_label = int(cls_pred.argmax().item())
            # Convert to (x1,y1,x2,y2) on original image scale (640 assumed)
            x1 = (px - pw/2) * 640
            y1 = (py - ph/2) * 640
            x2 = (px + pw/2) * 640
            y2 = (py + ph/2) * 640
            all_preds.append({
                "boxes": torch.tensor([[x1, y1, x2, y2]]),
                "scores": torch.tensor([pobj]),
                "labels": torch.tensor([pred_label]),
            })
            # Ground truth (scale to pixel coords)
            gt_box = boxes[0] * 640
            all_targets.append({
                "boxes": gt_box.unsqueeze(0),
                "labels": labels[0].unsqueeze(0),
            })

    if MeanAveragePrecision is None:
        raise ImportError("Install torchmetrics (pip install torchmetrics) for evaluation.")
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(all_preds, all_targets)
    results = metric.compute()
    print(f"Precision (AP@0.5): {results['map_50']:.4f}")
    print(f"mAP@0.5:0.95: {results['map']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    # Params and FLOPs
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    try:
        from torch.profiler import profile, ProfilerActivity
        dummy = torch.randn(1,3,640,640).to(device)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            model(dummy)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    except Exception as e:
        print("FLOPs profiling unavailable:", e)

if __name__ == "__main__":
    data_root = "/path/to/URPC2021"  # same root
    evaluate_model(data_root)
