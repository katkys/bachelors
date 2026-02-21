import os
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
import csv
import json
import face_detection as fd

TRUE_BBOXES_FILE = "" #path to the ground-truth-bboxes csv file -> see "ground_truth_bboxes_no_ears.csv" for the exact expected format
IMAGES_DIR_PATH = "" #path to directory containing image samples chosen for face-detection testing

def compute_iou(bbox1, bbox2):
    #bbox needs to be in format (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    a1 = max(x1, x3)
    b1 = max(y1, y3)
    a2 = min(x2, x4)
    b2 = min(y2, y4)

    intersection_area = max(0, a2 - a1) * max(0, b2 - b1)
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x4 - x3) * (y4 - y3)
    union_area = area_bbox1 + area_bbox2 - intersection_area

    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def evaluate_detector(output_dir, detector_name):
    os.makedirs(output_dir, exist_ok=True)
    input_dir = Path(IMAGES_DIR_PATH)
    output_dir = Path(output_dir)
    
    #load ground truth bboxes, which will be used for IoU computation
    true_bboxes = {}
    with open(TRUE_BBOXES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            shape_attrs = json.loads(row["region_shape_attributes"])

            x = shape_attrs["x"]
            y = shape_attrs["y"]
            width = shape_attrs["width"]
            height = shape_attrs["height"]

            true_bboxes[filename] = (x, y, x+width, y+height)

    #run face detection for each detector, compute IoU with ground truth, and save results to the output evaluation file
    eval_results_file_path = Path(output_dir) / f"eval_{detector_name}_conf{fd.MIN_CONFIDENCE}.txt"
    per_image_iou_csv_path = Path(output_dir) / f"{detector_name}_conf{fd.MIN_CONFIDENCE}_per_image_iou.csv"

    print(f"Started evaluation of {detector_name} face detector...")
    detected_bboxes = {}
    detector = fd.create_detector(detector_name)
    no_detection_count = 0
    detector_error_count = 0
    failed_load_count = 0
    
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(input_dir)
        out_subdir = output_dir / rel_path
        out_subdir.mkdir(parents=True, exist_ok=True)

        
        for fname in files:
            if not fname.lower().endswith(fd.VALID_EXTS):
                continue
            
            img_path = root_path / fname
            detected_bboxes[img_path.name] = None
            
            img = cv.imread(str(img_path))
            if img is None:
                failed_load_count += 1
                continue

            detections = None
            try:
                detections = detector.detect(img_path)
            except Exception as e:
                print(f"detector error for image '{img_path}': {e}")
                detector_error_count += 1
                continue
            
            if not detections:
                detected_bboxes[img_path.name] = None
                no_detection_count += 1
                continue

            # if multiple faces detected for image, only the one with highest confidence is considered
            # (images in the subset used for face-detection evaluation contain only one face each)
            max_score = -1
            best_detection = None
            for detection in detections:
                if detector_name == 'cv' and best_detection == None:
                    best_detection = detection

                if detection[4] > max_score and detection[4] >= fd.MIN_CONFIDENCE:
                    max_score = detection[4]
                    best_detection = detection

            if best_detection is None:
                no_detection_count += 1
                continue
            
            x1, y1, x2, y2, score = best_detection
            h, w, _ = img.shape
            x1, y1, x2, y2 = fd.clip_bbox(x1, y1, x2, y2, w, h)
            detected_bboxes[img_path.name] = (x1, y1, x2, y2)

    if detector_name == 'mediapipe':
        detector.close()

    #compute IoU for detected bboxes with ground truth
    ious = {}
    for img_path, detected_bbox in detected_bboxes.items():
        true_bbox = true_bboxes.get(img_path)
        if true_bbox is None:
            continue

        if detected_bbox is None:
            ious[img_path] = 0.0
            continue

        ious[img_path] = compute_iou(detected_bbox, true_bbox)

    mean_iou_all = sum(ious.values()) / len(ious)

    ious_detected_list = [iou for iou in ious.values() if iou > 0]
    if ious_detected_list:
        mean_iou_detected = sum(ious_detected_list) / len(ious_detected_list)
    else:   
        mean_iou_detected = 0.0

    print(f"\nFinished evaluation.\n")

    print(f"Writing evaluation results to {eval_results_file_path}...")
    with open(eval_results_file_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"Detector: {detector_name}\n")
        f.write(f"Confidence threshold: {fd.MIN_CONFIDENCE}\n")
        f.write(f"Total images processed: {len(detected_bboxes)}\n")
        f.write(f"Images with no detections: {no_detection_count}\n")
        f.write(f"Images failed to load: {failed_load_count}\n")
        f.write(f"Images for which detector error occurred: {detector_error_count}\n")
        f.write("\nRESULTS:\n")
        f.write(f"Mean IoU detected: {mean_iou_detected:.4f}\n")
        f.write(f"Mean IoU all (failed detections are given IoU=0): {mean_iou_all:.4f}\n")
    print(f"Evaluation results were saved.\n")

    print(f"Writing per-image IoU results to CSV file {per_image_iou_csv_path}...")
    with open(per_image_iou_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "iou"])
        for img, iou in ious.items():
            writer.writerow([img, f"{iou:.4f}"])
    print(f"Per-image IoU results were saved.\n")

def main():
    parser = ArgumentParser(description="Evaluate face detector based on IoU.")
    parser.add_argument("--dst", required=True, help="Path to the directory where output files will be saved.")
    parser.add_argument("--detector", required=True, choices=fd.DETECTOR_OPTIONS, help=f"Face detector to evaluate (options={fd.DETECTOR_OPTIONS})")
    args = parser.parse_args()

    evaluate_detector(args.dst, args.detector)

if __name__ == "__main__":
    main()