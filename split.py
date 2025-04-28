#!/usr/bin/env python3
"""
prepare_and_split_hardcoded.py

Reads:
    RAW_ROOT/images/*.jpg
    RAW_ROOT/annotations/*.xml

Writes to:
    OUT_ROOT/
      images/train  images/val  images/test
      labels/train  labels/val  labels/test

Splits 70/20/10 and converts each XML (VOC) → YOLO txt.
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

    imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith(".jpg"))
    print(f"[1] Found {len(imgs)} images in {img_src}")

    # Convert XML → YOLO txt (in‐memory)
    temp = {}
    for im in imgs:
        name = os.path.splitext(im)[0]
        xmlp = os.path.join(ann_src, name + ".xml")
        if not os.path.isfile(xmlp):
            continue
        lines = xml_to_yolo(xmlp)
        if lines:
            temp[name] = "\n".join(lines)

    print(f"[2] Parsed {len(temp)} XMLs with objects")

    # Split 70/20/10
    random.shuffle(imgs)
    N = len(imgs)
    n1 = int(0.7 * N)
    n2 = int(0.2 * N)
    splits = {
        "train": imgs[:n1],
        "val":   imgs[n1:n1+n2],
        "test":  imgs[n1+n2:],
    }

    # Make output dirs
    for s in splits:
        os.makedirs(os.path.join(out_root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", s), exist_ok=True)

    # Copy images and write labels
    for s, group in splits.items():
        ci, cl = 0, 0
        for im in group:
            name = os.path.splitext(im)[0]
            # Copy image
            shutil.copyfile(
                os.path.join(img_src, im),
                os.path.join(out_root, "images", s, im)
            )
            ci += 1
            # Write label if exists
            if name in temp:
                with open(os.path.join(out_root, "labels", s, name + ".txt"), "w") as f:
                    f.write(temp[name] + "\n")
                cl += 1
        print(f"[{s}] images: {ci}, labels: {cl}")

    print(f"[DONE] Prepared dataset at {out_root}/images and {out_root}/labels")

if __name__ == "__main__":
    prepare(RAW_ROOT, OUT_ROOT)

