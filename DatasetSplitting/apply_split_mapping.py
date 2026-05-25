from argparse import ArgumentParser
from pathlib import Path
import shutil
import csv
from collections import defaultdict

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

SPLIT_MAPPING_FILE_PATH = "./SplitMappingFilesCSV/split_mapping_5fold_cv_A.csv"
FINAL_SPLIT_CSV_PATH = "./SplitMappingFilesCSV/final_train_val_split_mapping_A.csv"


def collect_img_groups(src_path):
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


def load_cv_split_mapping(csv_mapping_path, k):
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


def load_final_split_mapping(csv_mapping_path):
    train_keys = set()
    val_keys = set()

    with open(str(csv_mapping_path), "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["artist"], row["base"])
            if row["split"] == "train":
                train_keys.add(key)
            elif row["split"] == "val":
                val_keys.add(key)

    return train_keys, val_keys


def copy_img_group(img_files, artist, dst_root):
    artist_dir = dst_root / artist
    artist_dir.mkdir(parents=True, exist_ok=True)

    for img_file in img_files:
        shutil.copy2(img_file, artist_dir / img_file.name)


def build_cv_dataset(groups, test_keys, fold_roles, dst_path, k):
    test_dir = dst_path / "test"

    for key in test_keys:
        if key not in groups:
            continue
        artist, img_base = key
        copy_img_group(groups[key], artist, test_dir)

    for fold in range(k):
        fold_dir = dst_path / f"fold_{fold+1}"
        train_dir = fold_dir / "train"
        val_dir = fold_dir / "val"

        for key, roles in fold_roles.items():
            if key not in groups:
                continue

            artist, img_base = key
            role = roles[fold]

            if role == "train":
                copy_img_group(groups[key], artist, train_dir)
            elif role == "val":
                copy_img_group(groups[key], artist, val_dir)


def build_final_train_val_dataset(groups, train_keys, val_keys, dst_path):
    train_dir = dst_path / "train"
    val_dir = dst_path / "val"

    for key in train_keys:
        if key not in groups:
            continue
        artist, _ = key
        copy_img_group(groups[key], artist, train_dir)

    for key in val_keys:
        if key not in groups:
            continue
        artist, _ = key
        copy_img_group(groups[key], artist, val_dir)


def __main__():
    parser = ArgumentParser(description="Apply both 5-fold CV split and final train/val split in one run.")
    parser.add_argument("--src", required=True, help="Path to the original unsplit dataset where images are organized in artist folders.")
    parser.add_argument("--dst", required=True, help="Path to the destination directory for the dataset.")
    parser.add_argument("--cv_csv", default=SPLIT_MAPPING_FILE_PATH, help="CSV file mapping for the 5-fold CV split.")
    parser.add_argument("--final_csv", default=FINAL_SPLIT_CSV_PATH, help="CSV file mapping for the final train/val split.")
    parser.add_argument("-k", type=int, default=5, help="Number of folds for cross-validation.")
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    if dst_path.exists():
        remove_existing_dst = input(f"Destination path '{dst_path}' already exists. Do you want to remove it and continue? (y/n): ")
        if remove_existing_dst.lower() == 'n':
            return
        shutil.rmtree(dst_path)

    dst_path.mkdir(parents=True, exist_ok=True)

    print("\nCollecting image groups...")
    img_groups = collect_img_groups(src_path)

    print("Loading 5-fold CV split mapping from CSV...")
    test_set, train_val_folds = load_cv_split_mapping(args.cv_csv, args.k)

    print("Building 5-fold CV dataset...")
    build_cv_dataset(img_groups, test_set, train_val_folds, dst_path, args.k)

    print("Loading final train/val split mapping from CSV...")
    train_keys, val_keys = load_final_split_mapping(args.final_csv)

    print("Building final train/val directories...")
    build_final_train_val_dataset(img_groups, train_keys, val_keys, dst_path)

    print("Done.")


if __name__ == "__main__":
    __main__()