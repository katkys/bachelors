import os
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
import numpy as np
import json
import csv
import face_detection as fd

PATH_TO_CSV_LANDMARKS = "manual_landmarks_for_failed_images.csv"

CROP_PADDING = 0.05
MIN_CROP_SIZE = 50

def save_failed_img(img, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(save_path), img)

def is_valid_crop(crop):
    h, w, _ = crop.shape
    return h >= MIN_CROP_SIZE and w >= MIN_CROP_SIZE and crop.size > 0

def extract_faces(src, dst):
    src = Path(src)
    dst = Path(dst)      
    os.makedirs(dst, exist_ok=True)

    small = 0
    landmarks_dict = {}

    print(f"\nLoading face landmarks from CSV file '{PATH_TO_CSV_LANDMARKS}'...")
    with open(PATH_TO_CSV_LANDMARKS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row["filename"]
            shape_attrs = json.loads(row["region_shape_attributes"])

            xs = shape_attrs["all_points_x"]
            ys = shape_attrs["all_points_y"]

            landmarks = np.array(list(zip(xs, ys)), dtype=np.int32)

            landmarks_dict[filename] = landmarks

    print(f"Loaded landmarks for {len(landmarks_dict)} images.")

    print(f"\nExtracting masked face-crops from images in '{src}'...")
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel_path = root_path.relative_to(src)
        out_subdir = dst / rel_path
        out_subdir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(fd.VALID_EXTS):
                continue

            img_path = root_path / fname
            img = cv.imread(str(img_path))
            if img is None:
                print(f"Failed to read: {img_path}")
                continue

            landmarks_np_array = landmarks_dict.get(fname)
            if landmarks_np_array is None:
                print(f"No landmarks found for image: {img_path}")
                continue
            
            h, w, _ = img.shape
            hull= cv.convexHull(landmarks_np_array)

            x, y, bw, bh = cv.boundingRect(hull)
            x1, y1, x2, y2 = fd.add_padding_to_bbox(x, y, x+bw, y+bh, CROP_PADDING)
            x1, y1, x2, y2 = fd.clip_bbox(x1, y1, x2, y2, w, h)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv.fillConvexPoly(mask, hull, 255)

            masked_img = cv.bitwise_and(img, img, mask=mask)
            face_crop = masked_img[y1:y2, x1:x2]

            if not is_valid_crop(face_crop):
                small += 1
                print(f"Detected face was too small in image: {img_path}")
                continue

            out_path = out_subdir / img_path.name
            cv.imwrite(str(out_path), face_crop)
    print("Process completed.")

    print(f"Number of images discarded due to insufficient face crop size: {small}")

def main():
    parser = ArgumentParser(description="Extract masked face-crops (with black background) from images using landmarks saved in a CSV file.")
    parser.add_argument("--src", required=True, help="Path to the input directory containing images whose landmarks are described in the CSV file.")
    parser.add_argument("--dst", required=True, help="Path to the output directory where processed images will be saved.")
    args = parser.parse_args()

    extract_faces(args.src, args.dst)

if __name__ == "__main__":
    main()
