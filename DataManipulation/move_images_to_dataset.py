import shutil
from pathlib import Path
from argparse import ArgumentParser

def restore_images(src, dst):
    src = Path(src)
    dst = Path(dst)

    moved = 0
    skipped = 0

    print("\nMoving images to their correct place in the dataset...")
    for img in src.iterdir():
        if not img.is_file():
            continue

        file_name = img.name
        try:
            split, author, filename = file_name.split("__", 2)
        except ValueError:
            print(f"Skipping (invalid filename): {file_name}")
            skipped += 1
            continue

        target_dir = dst / split / author
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        if target_path.exists():
            print(f"Skipping (already included in dataset): {target_path}")
            skipped += 1
            continue

        shutil.move(str(img), str(target_path))
        print(f"Moved {file_name} to {target_path}")
        moved += 1

    print(f"\nMoved {moved} images.")
    print(f"Skipped {skipped} files.")


def main():
    parser = ArgumentParser(help="Move images from a folder to their correct place in the dataset directory based on their name (format: 'split__author__fname').")
    parser.add_argument("--src", required=True, help="Directory with images which are to be moved to the dataset.")
    parser.add_argument("--dst", required=True, help="Main dataset directory.")
    args = parser.parse_args()

    restore_images(args.src, args.dst)


if __name__ == "__main__":
    main()
