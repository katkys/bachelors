import os
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
import logging
import numpy as np
import face_detection as fd

CROP_PADDING = 0.05
MIN_FACE_CROP_SIZE = 50 ** 2

def extract_faces(input_dir, output_dir, logger):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    failed_dir = output_dir / "manual_crops_needed"      
    failed_dir.mkdir(parents=True, exist_ok=True)        
    detector = fd.create_detector('mediapipe')

    success, failure, small = 0, 0, 0

    print(f"\nExtracting masked face-crops using MediaPipe Face Landmarker...")
    print(("NOTE: face landmark detection will work best if the input images are FACE CROPS "
           "with 25% PADDING added to the face bounding box before cropping. "
           "Resizing to 256x256px happens internally, but to preserve ratios input has to be of SQUARE SIZE.\n"))
    
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(input_dir)
        failed_subdir = failed_dir / rel_path
        out_subdir = output_dir / rel_path
        out_subdir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(fd.VALID_EXTS):
                continue

            img_path = root_path / fname
            img = cv.imread(str(img_path))
            if img is None:
                failure += 1
                logger.error(f"Failed to read: {img_path}")
                continue

            
            landmarks_np_array = None
            normalize_landmarks = True
            try:
                landmarks_np_array = detector.detect_landmarks(str(img_path), normalized=normalize_landmarks)
            except Exception as e:
                failure += 1
                logger.error(f"Detector error ({e}): {img_path}")
                failed_subdir.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(failed_subdir / img_path.name), img)
                continue
            
            if len(landmarks_np_array) == 0:
                failure += 1
                logger.info(f"No face detected in image: {img_path}")
                failed_subdir.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(failed_subdir / img_path.name), img)
                continue
            
            h, w, _ = img.shape
            if normalize_landmarks:
                landmarks_np_array = (landmarks_np_array * np.array([w, h])).astype(np.int32)

            hull= cv.convexHull(landmarks_np_array)

            x, y, bw, bh = cv.boundingRect(hull)
            x1, y1, x2, y2 = fd.add_padding_to_bbox(x, y, x+bw, x+bh, CROP_PADDING)
            x1, y1, x2, y2 = fd.clip_bbox(x1, y1, x2, y2, w, h)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv.fillConvexPoly(mask, hull, 255)

            masked_img = cv.bitwise_and(img, img, mask=mask)
            face_crop = masked_img[y1:y2, x1:x2]

            if face_crop.size < MIN_FACE_CROP_SIZE:
                small += 1
                logger.info(f"Detected face was too small in image: {img_path}")
                continue

            out_path = out_subdir / img_path.name
            cv.imwrite(str(out_path), face_crop)

            success += 1

    detector.close()

    print("Process completed.")
    log_file_path = output_dir / "face_extraction.log"
    print("\nSUMMARY:")
    print(f"Successfully processed {success} images.")
    print(f"Failed to detect face landmarks in {failure} images.")
    print(f"Discarded {small} images due to insufficient face crop size (<{MIN_FACE_CROP_SIZE}px).")
    print(f"See log file for details -> ({log_file_path})")

def main():
    parser = ArgumentParser(description="Extract masked face-crops (with black background) from images using MediaPipe Face Landmarker.")
    parser.add_argument("--src", required=True, help="Path to the input directory containing images you wish to process.")
    parser.add_argument("--dst", required=True, help="Path to the output directory where processed images will be saved.")
    args = parser.parse_args()

    output_dir = Path(args.dst)
    output_dir.mkdir(parents=True, exist_ok=True) 
    log_file_path = output_dir / "face_extraction.log"  
    logging.basicConfig(filename=str(log_file_path), encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)

    extract_faces(args.src, args.dst, logger)

if __name__ == "__main__":
    main()
