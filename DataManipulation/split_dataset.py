from argparse import ArgumentParser
import os
import shutil
from pathlib import Path
import random

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

def copy_and_split_dataset(src_dir, dst_dir):
    dst_path = Path(dst_dir)
    os.makedirs(dst_path, exist_ok=True)

    for subdir_name in ['training', 'validation', 'test']:
        subdir_path = dst_path / subdir_name
        os.makedirs(subdir_path, exist_ok=True)

    for artist_folder in os.listdir(src_dir):
        artist_path = Path(src_dir) / artist_folder
        if not artist_path.is_dir():
            continue

        images = [f for f in os.listdir(artist_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        total_images = len(images)

        train_end = int(total_images * TRAIN_RATIO)
        val_end = train_end + int(total_images * VAL_RATIO)

        splits = {
            'training': images[:train_end],
            'validation': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split_name, split_images in splits.items():
            split_artist_path = dst_path / split_name / artist_folder
            os.makedirs(split_artist_path, exist_ok=True)

            for image_name in split_images:
                src_image_path = artist_path / image_name
                dst_image_path = split_artist_path / image_name
                try:
                    shutil.copy2(src_image_path, dst_image_path)
                except Exception as e:
                    print(f"Error copying {src_image_path} to {dst_image_path}: {e}")
        

def main():
    parser = ArgumentParser(description="Copy dataset to given location and split it into train, validation and test datasets.")
    parser.add_argument("--src", type=str, required=True, help="Source folder with images organized in artist folders.")
    parser.add_argument("--dst", type=str, required=True, help="Destination folder for split dataset.")
    args = parser.parse_args()
    copy_and_split_dataset(args.src, args.dst)

if __name__ == "__main__":
    main()
