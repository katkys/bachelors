import os
from argparse import ArgumentParser
from pathlib import Path
import cv2 as cv
import logging
import face_detection as fd

PADDING = 0.05
MIN_FACE_CROP_SIZE = 50

def extract_faces(input_dir, output_dir, logger, detector_name):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if detector_name not in fd.DETECTOR_OPTIONS:
        raise ValueError(f"Unsupported detector: {detector_name}")
    detector = fd.create_detector(detector_name)

    success, failure = 0, 0

    print(f"Extracting faces using {detector_name} face detector...")
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(input_dir)
        out_subdir = output_dir / rel_path
        out_subdir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(fd.VALID_EXTS):
                continue

            img_path = root_path / fname
            img = cv.imread(str(img_path))
            if img is None:
                failure += 1
                logger.error(f"Failed to read image: {img_path}")
                continue

            detections = None
            try:
                detections = detector.detect(img_path)
            except Exception as e:
                failure += 1
                logger.error(f"Detector error for image {img_path}: {e}")
                continue
            
            face_found = False
            face_idx = 1
            for detection in detections:
                x1, y1, x2, y2, score = detection
                if score < fd.MIN_CONFIDENCE and detector_name != 'opencv': #opencv doesn't provide confidence scores and so our OpenCVDetector.detect method sets it to -1.0
                    continue
                
                h, w, _ = img.shape
                x1, y1, x2, y2 = fd.add_padding_to_bbox(x1, y1, x2, y2, PADDING)
                x1, y1, x2, y2 = fd.clip_bbox(x1, y1, x2, y2, w, h)

                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0 or face_crop.shape[0] < MIN_FACE_CROP_SIZE or face_crop.shape[1] < MIN_FACE_CROP_SIZE:
                    continue

                out_fname = f"{img_path.stem}_face{face_idx}.jpg"
                out_path = out_subdir / out_fname
                cv.imwrite(str(out_path), face_crop)

                face_found = True
                face_idx += 1

            if not face_found:
                failure += 1
                logger.info(f"No faces detected in image: {img_path}")
            else:
                success += 1

    if detector_name == 'mediapipe':
        detector.close()

    print("Face extraction completed.")
    log_file_path = output_dir / "face_extraction.log"
    print("\nSUMMARY:")
    print(f"Successfully processed {success} images.")
    print(f"Failed to extract faces from {failure} images.")
    print(f"See log file for details -> ({log_file_path})")

def main():
    parser = ArgumentParser(description="Extract face crops from images using chosen face-detection method.")
    parser.add_argument("--src", required=True, help="Path to the input directory containing images organized in artist subdirectories.")
    parser.add_argument("--dst", required=True, help="Path to the output directory where extracted face crops will be saved.")
    parser.add_argument("--detector", required=True, choices=fd.DETECTOR_OPTIONS, help=f"Face detection model to use (options: {fd.DETECTOR_OPTIONS})")
    args = parser.parse_args()

    output_dir = Path(args.dst)
    output_dir.mkdir(parents=True, exist_ok=True) 
    log_file_path = output_dir / "face_extraction.log"  
    logging.basicConfig(filename=str(log_file_path), encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)

    extract_faces(args.src, args.dst, logger, args.detector)

if __name__ == "__main__":
    main()
