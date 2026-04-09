import os
from argparse import ArgumentParser
from pathlib import Path
import random
import csv
from collections import defaultdict


IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
VAL_RATIO = 0.15 # for final model training, we reduce the val ratio to 15% (in the 5-fold cv training we used 20%)

#  we want to keep the same test set as in the 5-fold cv,
#  so we will read the groups from the 5-fold cv csv mapping file,
#  and only randomly split the train_val groups into train/val for the final model training
def collect_train_val_groups_from_cv_csv(cv_csv_path):
    groups = []

    with open(cv_csv_path, newline='') as f:  
        reader = csv.DictReader(f)  
        for row in reader:
            if row['split'] == 'train_val': 
                artist = row['artist']
                base = row['base']
                groups.append((artist, base))
    return groups


def stratified_split(img_groups, val_ratio):
    artist_to_keys = defaultdict(list)
    for artist, base in img_groups:
        artist_to_keys[artist].append((artist, base))

    train = []
    val = []

    for artist, keys in artist_to_keys.items():
        random.shuffle(keys)
        
        val_count = max(1, int(len(keys) * val_ratio))
        val.extend(keys[:val_count])
        train.extend(keys[val_count:])

    return train, val


def build_csv(groups, val_keys, output_path):
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
    parser = ArgumentParser(description="Create final train/val split CSV from 5-fold CV mapping CSV.")  
    parser.add_argument("--cv_csv", required=True, help="CSV file from 5-fold CV mapping containing train_val/test info.")  
    parser.add_argument("--dst", required=True, help="Output directory path for the final train/val CSV split mapping file.")

    args = parser.parse_args()


    print("\nCollecting images...")
    groups = collect_train_val_groups_from_cv_csv(args.cv_csv)

    print("Creating stratified train/val split...")
    train, val = stratified_split(groups, VAL_RATIO)

    print("Writing CSV...")
    build_csv(groups, val, args.dst)

    print("Done.")


if __name__ == "__main__":
    main()
