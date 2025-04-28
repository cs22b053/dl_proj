#!/usr/bin/env python3
"""
prepare_and_split_hardcoded.py

Reads from:
    RAW_ROOT/images/*.jpg
    RAW_ROOT/annotations/*.xml  (VOC format)
    RAW_ROOT/annotations/*.txt  (YOLO format)

Writes to:
    OUT_ROOT/
      images/train  images/val  images/test
      labels/train  labels/val  labels/test

Splits 70/20/10.
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET

# ======== CONFIGURE PATHS HERE ========
RAW_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021/urpc2020"
OUT_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021_prepared"
# =========================================

# The 5 URPC classes in order
CLASSES = ["echinus", "holothurian", "starfish", "scallop", "water weeds"]

def xml_to_yolo(xml_path):
    """Parse VOC XML at xml_path → list of YOLO-format strings."""
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    out = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASSES:
            continue
        cid = CLASSES.index(name)
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        x_c = (xmin + xmax) / 2 / w
        y_c = (ymin + ymax) / 2 / h
        bw  = (xmax - xmin) / w
        bh  = (ymax - ymin) / h
        out.append(f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    return out

def prepare(raw_root, out_root):
    img_src = os.path.join(raw_root, "images")
    ann_src = os.path.join(raw_root, "annotations")
    assert os.path.isdir(img_src), f"Missing images folder: {img_src}"
    assert os.path.isdir(ann_src), f"Missing annotations folder: {ann_src}"

    # 1) List images
    imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith(".jpg"))
    print(f"[1] Found {len(imgs)} images")

    # 2) Read all annotations (txt or xml)
    ann_map = {}
    for fn in os.listdir(ann_src):
        base, ext = os.path.splitext(fn)
        path = os.path.join(ann_src, fn)
        if ext.lower() == ".txt":
            # YOLO format: copy lines
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                ann_map[base] = lines
        elif ext.lower() == ".xml":
            # VOC xml: parse
            lines = xml_to_yolo(path)
            if lines:
                ann_map[base] = lines

    print(f"[2] Loaded annotations for {len(ann_map)} images")

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

    # 4) Make dirs
    for s in splits:
        os.makedirs(os.path.join(out_root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", s), exist_ok=True)

    # 5) Copy images + write labels
    for s, group in splits.items():
        ci = cl = 0
        for im in group:
            base = os.path.splitext(im)[0]
            # copy image
            shutil.copyfile(
                os.path.join(img_src, im),
                os.path.join(out_root, "images", s, im)
            ); ci += 1
            # write label if we have it
            if base in ann_map:
                lbl_path = os.path.join(out_root, "labels", s, base + ".txt")
                with open(lbl_path, "w") as f:
                    f.write("\n".join(ann_map[base]) + "\n")
                cl += 1
        print(f"[{s}] images: {ci}, labels: {cl}")

    print(f"[DONE] Dataset prepared under\n  {out_root}/images\n  {out_root}/labels")

if __name__ == "__main__":
    prepare(RAW_ROOT, OUT_ROOT)

