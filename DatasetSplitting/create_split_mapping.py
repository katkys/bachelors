import os
from argparse import ArgumentParser
from pathlib import Path
import random
import csv
from collections import defaultdict

RANDOM_SEED = 27

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

K = 5
TEST_RATIO = 0.15
VAL_RATIO = 0.2


def collect_img_groups(src_dir):
    groups = []

    for artist_dir in Path(src_dir).iterdir():
        if not artist_dir.is_dir():
            continue

        for file in artist_dir.iterdir():
            if file.suffix.lower() not in IMAGE_EXTS:
                continue

            base = file.stem
            groups.append((artist_dir.name, base))

    return groups


def stratified_split(groups, split_ratio):
    artist_to_keys = defaultdict(list)
    for artist, base in groups:
        artist_to_keys[artist].append((artist, base))

    split_a = []
    split_b = []

    for artist, keys in artist_to_keys.items():
        random.shuffle(keys)
        split_count = max(1, int(len(keys) * split_ratio))

        split_b.extend(keys[:split_count])
        split_a.extend(keys[split_count:])

    return split_a, split_b


def create_folds(train_val_keys, k):
    artist_to_keys = defaultdict(list)

    for key in train_val_keys:
        artist = key[0]
        artist_to_keys[artist].append(key)

    fold_assignment = {}
    for artist, keys in artist_to_keys.items():
        random.shuffle(keys)
        for i, key in enumerate(keys):
            fold_assignment[key] = i % k

    return fold_assignment


def build_cv_csv(groups, test_keys, fold_assignment, k, output_path):
    test_set = set(test_keys)
    header = ["artist", "base", "split"] + [f"fold_{i+1}" for i in range(k)]
    os.makedirs(output_path, exist_ok=True)
    output_csv_file_path = Path(output_path) / "split_mapping_5fold_cv.csv"

    with open(str(output_csv_file_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for key in groups:
            artist, base = key

            if key in test_set:
                row = [artist, base, "test"] + ["-"] * k
            else:
                assigned_fold = fold_assignment[key]

                fold_roles = []
                for i in range(k):
                    if i == assigned_fold:
                        fold_roles.append("val")
                    else:
                        fold_roles.append("train")
                row = [artist, base, "train_val"] + fold_roles

            writer.writerow(row)


def build_final_train_val_csv(groups, val_keys, output_path):
    val_set = set(val_keys)
    header = ["artist", "base", "split"]
    os.makedirs(output_path, exist_ok=True)
    output_csv_file_path = Path(output_path) / "final_train_val_split_mapping.csv"

    with open(str(output_csv_file_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for key in groups:
            artist, base = key

            if key in val_set:
                row = [artist, base, "val"]
            else:
                row = [artist, base, "train"]

            writer.writerow(row)


def main():
    parser = ArgumentParser(description="Create split mapping CSV files (5-fold CV, final train/val).")
    parser.add_argument("--src", required=True, help="Main directory where images are organized in artist folders.")
    parser.add_argument("--dst", required=True, help="Output directory path for generated CSV files.")
    parser.add_argument("-k", type=int, default=5)

    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    print("\nCollecting images...")
    groups = collect_img_groups(args.src)

    print("Creating stratified train+val/test split...")
    train_val, test = stratified_split(groups, TEST_RATIO)

    print(f"Creating {args.k} folds from train+val...")
    fold_assignment = create_folds(train_val, args.k)

    print("Creating stratified train/val split...")
    train, val = stratified_split(train_val, VAL_RATIO)

    print("Writing 5-fold CV CSV...")
    build_cv_csv(groups, test, fold_assignment, args.k, args.dst)

    print("Writing final train/val CSV...")
    build_final_train_val_csv(train_val, val, args.dst)

    print("Done.")


if __name__ == "__main__":
    main()