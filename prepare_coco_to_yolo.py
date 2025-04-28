#!/usr/bin/env python3
"""
prepare_coco_to_yolo.py

Reads COCO JSON (instances_train.json, instances_test.json) 
and creates YOLO labels + splits into train/val/test folders.

Hard-coded paths to match your cluster.
"""

import os
import json
import random
import shutil
from collections import defaultdict

# ====== CONFIGURE PATHS ======
RAW_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021/urpc2020"
OUT_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021_prepared"
TRAIN_JSON = os.path.join(RAW_ROOT, "annotations", "instances_train.json")
TEST_JSON  = os.path.join(RAW_ROOT, "annotations", "instances_test.json")
IMG_SRC    = os.path.join(RAW_ROOT, "images")
# ===============================

# URPC2021 classes (fix mapping if needed)
CLASSES = ["echinus", "holothurian", "starfish", "scallop", "water weeds"]
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

def load_coco(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    imgs = {img["id"]: img for img in data["images"]}
    anns = defaultdict(list)
    for ann in data["annotations"]:
        anns[ann["image_id"]].append(ann)
    cats = {cat["id"]: cat["name"] for cat in data["categories"]}
    return imgs, anns, cats

def coco_ann_to_yolo(ann, img_w, img_h):
    """Convert COCO annotation dict to YOLO format line."""
    bbox = ann["bbox"]  # COCO: [x_min, y_min, width, height]
    x_center = (bbox[0] + bbox[2] / 2) / img_w
    y_center = (bbox[1] + bbox[3] / 2) / img_h
    w = bbox[2] / img_w
    h = bbox[3] / img_h
    return f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

def prepare():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Load train/test COCO JSON
    train_imgs, train_anns, train_cats = load_coco(TRAIN_JSON)
    test_imgs,  test_anns,  test_cats  = load_coco(TEST_JSON)

    all_imgs = list(train_imgs.values()) + list(test_imgs.values())
    all_anns = {**train_anns, **test_anns}

    print(f"[INFO] Total images = {len(all_imgs)}")
    print(f"[INFO] Categories (from JSON): {train_cats}")

    # 70/20/10 split
    random.shuffle(all_imgs)
    N = len(all_imgs)
    n1 = int(0.7 * N)
    n2 = int(0.2 * N)
    splits = {
        "train": all_imgs[:n1],
        "val":   all_imgs[n1:n1+n2],
        "test":  all_imgs[n1+n2:],
    }

    # Make folders
    for s in splits:
        os.makedirs(os.path.join(OUT_ROOT, "images", s), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", s), exist_ok=True)

    for split, img_list in splits.items():
        img_dir = os.path.join(OUT_ROOT, "images", split)
        lbl_dir = os.path.join(OUT_ROOT, "labels", split)

        img_count = 0
        lbl_count = 0
        for img in img_list:
            img_id = img["id"]
            img_name = img["file_name"]
            img_w = img["width"]
            img_h = img["height"]

            # Copy image
            src_img_path = os.path.join(IMG_SRC, img_name)
            dst_img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(src_img_path):
                continue
            shutil.copyfile(src_img_path, dst_img_path)
            img_count += 1

            # Create YOLO label
            label_lines = []
            for ann in all_anns.get(img_id, []):
                cat_name = train_cats.get(ann["category_id"], None) or test_cats.get(ann["category_id"], None)
                if cat_name not in CLASS_NAME_TO_ID:
                    continue
                cls_id = CLASS_NAME_TO_ID[cat_name]
                bbox = ann["bbox"]
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h
                label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            if label_lines:
                label_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))
                lbl_count += 1

        print(f"[{split}] images copied = {img_count}, labels written = {lbl_count}")

    print("[DONE] COCO → YOLO conversion and splitting completed.")

if __name__ == "__main__":
    prepare()

