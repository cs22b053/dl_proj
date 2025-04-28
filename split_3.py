#!/usr/bin/env python3
"""
prepare_and_split_hardcoded.py

Recursively finds VOC XMLs under RAW_ROOT/annotations, converts them to
YOLO .txt, and splits images 70/20/10 into OUT_ROOT/images/{train,val,test}
and OUT_ROOT/labels/{train,val,test}.

Edit RAW_ROOT and OUT_ROOT below to match your cluster paths.
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET

# ===== CONFIGURE PATHS BELOW =====
RAW_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021/urpc2020"
OUT_ROOT = "/scratch/cs22b053/dl/dl_proj/URPC2021_prepared"
# ==================================

CLASSES = ["echinus","holothurian","starfish","scallop","water weeds"]

def xml_to_yolo(xml_path):
    """Parse a VOC XML and return list of YOLO-format lines."""
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
    # 1) Locate your images & annotations
    img_src = os.path.join(raw_root, "images")
    ann_src = os.path.join(raw_root, "annotations")
    assert os.path.isdir(img_src), f"No images folder at {img_src}"
    assert os.path.isdir(ann_src), f"No annotations folder at {ann_src}"

    # 2) Find all JPG images
    imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith(".jpg"))
    print(f"[1] Found {len(imgs)} images in {img_src}")

    # 3) Recursively find all XMLs under annotations/
    annot_map = {}
    for dp, dn, files in os.walk(ann_src):
        for f in files:
            if f.lower().endswith(".xml"):
                key = os.path.splitext(f)[0]
                annot_map[key] = os.path.join(dp, f)
    print(f"[2] Found {len(annot_map)} XML annotation files (recursively)")

    # 4) Convert each XML → YOLO lines (in-memory)
    temp = {}
    for name, xmlp in annot_map.items():
        yolo_lines = xml_to_yolo(xmlp)
        if yolo_lines:
            temp[name] = "\n".join(yolo_lines)

    print(f"[3] Parsed {len(temp)} XMLs with ≥1 <object> into YOLO format")

    # 5) Split into train/val/test (70/20/10)
    random.shuffle(imgs)
    N = len(imgs)
    n1 = int(0.7 * N)
    n2 = int(0.2 * N)
    splits = {
        "train": imgs[:n1],
        "val":   imgs[n1:n1+n2],
        "test":  imgs[n1+n2:],
    }

    # 6) Make output directories
    for s in splits:
        os.makedirs(os.path.join(out_root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", s), exist_ok=True)

    # 7) Copy images and write labels
    for s, group in splits.items():
        ci, cl = 0, 0
        for im in group:
            name = os.path.splitext(im)[0]
            # copy image
            shutil.copyfile(
                os.path.join(img_src, im),
                os.path.join(out_root, "images", s, im)
            ); ci += 1
            # write label if we parsed it
            if name in temp:
                with open(os.path.join(out_root, "labels", s, name + ".txt"), "w") as f:
                    f.write(temp[name] + "\n")
                cl += 1
        print(f"[{s}] images copied: {ci}, labels written: {cl}")

    print(f"[DONE] Dataset prepared under\n  {out_root}/images/ and\n  {out_root}/labels/")

if __name__ == "__main__":
    prepare(RAW_ROOT, OUT_ROOT)

