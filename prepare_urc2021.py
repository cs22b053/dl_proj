#!/usr/bin/env python3
import os
import requests
import zipfile
import shutil
import xml.etree.ElementTree as ET
import random

# -------- CONFIGURATION --------
# URL of the URPC2021 dataset zip (change if needed to official link)
DATA_URL = "https://github.com/xiaoDetection/Learning-Heavily-Degraed-Prior/releases/download/datasets/urpc2020.zip"
# Local paths
DATA_DIR = "URPC2021"                 # Base folder for dataset
ZIP_PATH = "URPC2021.zip"             # Temporary download filename

# Class names in the dataset (order defines class index in YOLO labels)
CLASSES = ["holothurian", "echinus", "scallop", "starfish"]

# Train/Val/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1

# -------- STEP 1: DOWNLOAD DATASET ZIP --------
if not os.path.exists(ZIP_PATH):
    print(f"Downloading URPC2021 dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# -------- STEP 2: EXTRACT ZIP CONTENT --------
if not os.path.exists(DATA_DIR):
    print(f"Extracting {ZIP_PATH} to {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

# -------- STEP 3: LOCATE IMAGES AND ANNOTATIONS --------
# We assume images are .jpg files somewhere under DATA_DIR, and annotations may be .xml or .txt.
# Build list of image file paths:
image_paths = []
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            image_paths.append(os.path.join(root, file))
image_paths.sort()
print(f"Found {len(image_paths)} image files.")

# Create output directories for images and labels
for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(DATA_DIR, "images", subset), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "labels", subset), exist_ok=True)

# -------- STEP 4: CONVERT ANNOTATIONS TO YOLO FORMAT --------
def convert_annotation(img_path, label_output_path):
    """
    Parse annotation for one image and write YOLO-format labels.
    Supports Pascal VOC .xml files or skips if no annotation found.
    """
    # Determine annotation file: same basename but with .xml
    base = os.path.splitext(img_path)[0]
    xml_path = base + ".xml"
    if not os.path.exists(xml_path):
        return  # No annotation to convert
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = float(root.findtext('size/width'))
    img_height = float(root.findtext('size/height'))
    with open(label_output_path, "w") as out_file:
        for obj in root.findall('object'):
            cls = obj.findtext('name').strip()
            if cls not in CLASSES:
                continue
            cls_idx = CLASSES.index(cls)
            # Get bounding box and convert to YOLO normalized format
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.findtext('xmin'))
            ymin = float(bndbox.findtext('ymin'))
            xmax = float(bndbox.findtext('xmax'))
            ymax = float(bndbox.findtext('ymax'))
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            box_width = xmax - xmin
            box_height = ymax - ymin
            # Normalize to [0,1]
            x_center /= img_width
            y_center /= img_height
            box_width /= img_width
            box_height /= img_height
            out_file.write(f"{cls_idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

# Convert all annotations and store label file paths
converted_labels = {}
for img_path in image_paths:
    label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    # Temporary store labels in-memory mapping until splitting
    label_temp_path = os.path.join(DATA_DIR, label_name)
    convert_annotation(img_path, label_temp_path)
    # Only keep mapping if the label file was created
    if os.path.exists(label_temp_path):
        converted_labels[img_path] = label_temp_path

print(f"Converted {len(converted_labels)} annotations to YOLO format.")

# -------- STEP 5: SPLIT INTO TRAIN/VAL/TEST --------
random.shuffle(image_paths)
num_images = len(image_paths)
num_train = int(TRAIN_RATIO * num_images)
num_val   = int(VAL_RATIO * num_images)
# rest is test
num_test = num_images - num_train - num_val

train_images = image_paths[:num_train]
val_images   = image_paths[num_train:num_train+num_val]
test_images  = image_paths[num_train+num_val:]

print(f"Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images.")

# Helper to copy files to subset folders
def copy_to_subset(img_list, subset):
    for img_path in img_list:
        fname = os.path.basename(img_path)
        # Copy image
        dst_img = os.path.join(DATA_DIR, "images", subset, fname)
        shutil.copyfile(img_path, dst_img)
        # Copy corresponding label if it exists (converted above)
        if img_path in converted_labels:
            dst_lbl = os.path.join(DATA_DIR, "labels", subset, os.path.basename(converted_labels[img_path]))
            shutil.copyfile(converted_labels[img_path], dst_lbl)

# Perform copy for each split
copy_to_subset(train_images, "train")
copy_to_subset(val_images,   "val")
copy_to_subset(test_images,  "test")

print("Dataset has been organized into images/ and labels/ folders in train/val/test splits.")

# -------- CLEANUP (optional) --------
# Remove the downloaded zip file and any temporary label files if desired
# os.remove(ZIP_PATH)
# for lbl in converted_labels.values():
#     os.remove(lbl)

print("URPC2021 dataset preparation complete. Directory structure is ready for YOLO training.")
