from argparse import ArgumentParser
from pathlib import Path
import shutil
import csv
from collections import defaultdict

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
SPLIT_MAPPING_FILE_PATH = "./split_mapping_5fold_cv_Dataset_A.csv"


def collect_img_groups(src_path): #groups face crops derived from one original image (base image) together
    groups = defaultdict(list)

    for artist_dir in src_path.iterdir():
        if not artist_dir.is_dir():
            continue

        for file in artist_dir.iterdir():
            if file.suffix.lower() not in IMAGE_EXTS:
                continue

            base_img_name = file.stem.split("_face")[0]
            groups[(artist_dir.name, base_img_name)].append(file)
    
    return groups

def load_split_mapping(csv_mapping_path, k):
    test_set = set()
    fold_roles = {}

    with open(str(csv_mapping_path), "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            artist = row["artist"]
            img_base = row["base"]
            key = (artist, img_base)

            if row["split"] == "test":
                test_set.add(key)
            else:
                fold_roles[key] = [row[f"fold_{i+1}"] for i in range(k)]
    
    return test_set, fold_roles

def copy_img_group(img_files, artist, dst_root):
    artist_dir = dst_root / artist
    artist_dir.mkdir(parents=True, exist_ok=True)

    for img_file in img_files:
        shutil.copy2(img_file, artist_dir / img_file.name)

def build_dataset(groups, test_keys, fold_roles, dst_path, k):
    # test set directory
    test_dir = dst_path / "test"
    for key in test_keys:
        if key not in groups:
            continue  
        artist, img_base = key
        copy_img_group(groups[key], artist, test_dir)

    # fold directories
    for fold in range(k):
        fold_dir = dst_path / f"fold_{fold+1}"
        train_dir = fold_dir / "training"
        val_dir = fold_dir / "validation"

        for key, roles in fold_roles.items():
            if key not in groups:
                continue  

            artist, img_base = key
            role = roles[fold]

            if role == "train":
                copy_img_group(groups[key], artist, train_dir)
            elif role == "val":
                copy_img_group(groups[key], artist, val_dir)


def __main__():
    parser = ArgumentParser(description="Apply k-fold cross-validation split (with a held-out test set) based on a provided CSV mapping file.")
    parser.add_argument("--src", required=True, help="Path to the source directory where images are organzied in artist folders.")
    parser.add_argument("--dst", required=True, help="Path to the destination directory where newly created dataset will be stored.")
    parser.add_argument("-k", type=int, default=5, help="Number of folds for cross-validation (default: 5).")
    args= parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True)

    print("\nCollecting image groups...")
    img_groups = collect_img_groups(src_path)

    print("Loading split mapping from CSV...")
    test_set, train_val_folds = load_split_mapping(str(SPLIT_MAPPING_FILE_PATH), args.k)

    print("Building dataset directory structure and copying images...")
    build_dataset(img_groups, test_set, train_val_folds, dst_path, args.k)
    
    print("Done.")


if __name__ == "__main__":
    __main__()
