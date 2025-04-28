#!/usr/bin/env python3
"""
prepare_and_split_hardcoded.py

Hardcodes RAW_ROOT and OUT_ROOT, reads annotations as either:
 - VOC XMLs (parses & converts)
 - YOLO .txt files (copies directly)

Produces:
 OUT_ROOT/
   images/train  images/val  images/test
   labels/train  labels/val  labels/test
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET

# ====== CONFIGURE PATHS HERE ======
RAW_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021/urpc2020"
OUT_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021_prepared"
# ==================================

CLASSES = ["echinus","holothurian","starfish","scallop","water weeds"]

def xml_to_yolo(xml_path):
    """Parse VOC XML -> list of YOLO-format lines."""
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    out = []
    for obj in root.findall("object"):
        cls = obj.findtext("name")
        if cls not in CLASSES: 
            continue
        cid = CLASSES.index(cls)
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
        x_c = (xmin + xmax)/2 / w
        y_c = (ymin + ymax)/2 / h
        bw  = (xmax - xmin) / w
        bh  = (ymax - ymin) / h
        out.append(f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    return out

def prepare(raw_root, out_root):
    img_src = os.path.join(raw_root, "images")
    ann_src = os.path.join(raw_root, "annotations")
    assert os.path.isdir(img_src), f"Missing images folder: {img_src}"
    assert os.path.isdir(ann_src), f"Missing annotations folder: {ann_src}"

    # 1) List all images
    imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith(".jpg"))
    print(f"[1] Found {len(imgs)} images in {img_src}")

    # 2) Read annotations (XML or TXT)
    temp = {}  # base name -> list of YOLO lines
    for fn in os.listdir(ann_src):
        base, ext = os.path.splitext(fn)
        path = os.path.join(ann_src, fn)
        if ext.lower() == ".xml":
            lines = xml_to_yolo(path)
            if lines:
                temp[base] = lines
        elif ext.lower() == ".txt":
            # assume already YOLO-format: copy lines directly
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                temp[base] = lines
    print(f"[2] Loaded annotations for {len(temp)} images (XML or TXT)")

    # 3) Split 70/20/10
    random.shuffle(imgs)
    N = len(imgs)
    n1 = int(0.7 * N)
    n2 = int(0.2 * N)
    splits = {
        "train": imgs[:n1],
        "val":   imgs[n1:n1+n2],
        "test":  imgs[n1+n2:],
    }

    # 4) Create output dirs
    for s in splits:
        os.makedirs(os.path.join(out_root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", s), exist_ok=True)

    # 5) Copy images & write labels
    for s, group in splits.items():
        cnt_img, cnt_lbl = 0, 0
        for im in group:
            base = os.path.splitext(im)[0]
            # copy image
            shutil.copyfile(
                os.path.join(img_src, im),
                os.path.join(out_root, "images", s, im)
            )
            cnt_img += 1
            # write or copy label if available
            if base in temp:
                out_lbl = os.path.join(out_root, "labels", s, base + ".txt")
                with open(out_lbl, "w") as f:
                    f.write("\n".join(temp[base]) + "\n")
                cnt_lbl += 1
        print(f"[{s}] images: {cnt_img}, labels: {cnt_lbl}")

    print(f"[DONE] Prepared dataset under {out_root}/images & {out_root}/labels")

if __name__ == "__main__":
    prepare(RAW_ROOT, OUT_ROOT)

