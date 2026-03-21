from argparse import ArgumentParser
import shutil
from pathlib import Path
import random
from collections import defaultdict

RANDOM_SEED = 27
K = 5 
IMAGE_EXTS = ('.jpg', '.png', '.jpeg')
FACES_DATASET = True

def collect_grouped_samples(src_dirs): #puts faces extracted from one image to one group
    groups = defaultdict(list)

    for src_dir in src_dirs:
        for artist_dir in Path(src_dir).iterdir():
            if not artist_dir.is_dir():
                continue

            for file in artist_dir.iterdir():
                if file.suffix.lower() not in IMAGE_EXTS:
                    continue
                
                img_name = file.stem
                if FACES_DATASET:
                    base = img_name.split("_face")[0]
                else:
                    base = img_name
                
                key = (artist_dir.name, base)
                groups[key].append(file)
    
    return groups


def create_folds(groups, k):
    random.seed(RANDOM_SEED)

    artist_to_group_keys = defaultdict(list)
    for (artist, img_base) in groups.keys():
        artist_to_group_keys[artist].append((artist, img_base))

    folds = [[] for _ in range(k)]

    for artist, img_group_keys in artist_to_group_keys.items():
        random.shuffle(img_group_keys)

        for i, key in enumerate(img_group_keys):
            folds[i % k].append(key)

    return folds


def copy_group(group_files, artist, target_root):
    artist_dir = target_root / artist
    artist_dir.mkdir(parents=True, exist_ok=True)

    for file in group_files:
        shutil.copy2(file, artist_dir / file.name)


def build_folds(folds, groups, dst_path):
    for i in range(len(folds)):
        fold_dir = dst_path / f"fold_{i+1}"
        train_dir = fold_dir / "training"
        val_dir = fold_dir / "validation"

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        val_keys = folds[i]
        train_keys = [k for f in (folds[:i] + folds[i+1:]) for k in f]

        for key in val_keys:
            artist, _ = key
            copy_group(groups[key], artist, val_dir)

        for key in train_keys:
            artist, _ = key
            copy_group(groups[key], artist, train_dir)

def main():
    parser = ArgumentParser(description="Convert training/valiadtion/test dataset into k-fold cross-validation dataset (default k=5).")
    parser.add_argument("--src", required=True, help="Source dataset directory containing 'training', 'validation' and 'test' subdirectories.")
    parser.add_argument("--dst", required=True, help="Destination directory for the new dataset.")
    parser.add_argument("-k", default=5, type=int, choices=range(2, 11), help="The number of folds.")

    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    train_dir = src_path / "training"
    val_dir = src_path / "validation"
    test_dir = src_path / "test"

    if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
        raise ValueError("Expected 'training', 'validation', and 'test' folders in the main source folder.")

    if dst_path.exists():
        shutil.rmtree(dst_path)
        dst_path.mkdir(parents=True)

    print("\nCollecting all training and validation samples...")
    groups = collect_grouped_samples([train_dir, val_dir])

    K = args.k

    print(f"Creating {K} stratified folds...")
    folds = create_folds(groups, K)

    print("Building fold directories...")
    build_folds(folds, groups, dst_path)

    print("Copying test set...")
    dst_test = dst_path / "test"
    shutil.copytree(test_dir, dst_test)

    print(f"Done creating {K}-fold cross-validation dataset.")


if __name__ == "__main__":
    main()
