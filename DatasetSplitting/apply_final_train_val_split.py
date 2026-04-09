from argparse import ArgumentParser
from pathlib import Path
import shutil
import csv
from collections import defaultdict

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
FINAL_SPLIT_CSV_PATH = "./final_train_val_split_mapping.csv"  # path to train/val CSV mapping file


def collect_img_groups(src_path):  # groups face crops derived from one original image (base image) together
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
    parser = ArgumentParser(description="Create final train/val directories based on split CSV.")
    parser.add_argument("--src", required=True, help="Path to the original unsplit dataset with images organzied in artist folders.") 
    parser.add_argument("--dst", required=True, help="Path to the destination directory where train/val directories will be created.")
    parser.add_argument("--csv", default=FINAL_SPLIT_CSV_PATH, help="CSV file mapping of final train/val split.") 

    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    dst_path.mkdir(parents=True)

    print("\nCollecting image groups...")
    img_groups = collect_img_groups(src_path)

    print("Loading final train/val split mapping from CSV...")
    train_keys, val_keys = load_final_split_mapping(args.csv)  

    print("Building final train/val dataset directories and copying images...")
    build_final_train_val_dataset(img_groups, train_keys, val_keys, dst_path)  

    print("Done.")


if __name__ == "__main__":
    __main__()
