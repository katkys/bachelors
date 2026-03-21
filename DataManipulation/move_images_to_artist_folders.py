import shutil
from pathlib import Path
from argparse import ArgumentParser

def restore_images(src, dst):
    src = Path(src)
    dst = Path(dst)

    moved = 0
    skipped = 0

    print("\nMoving images to the correct artist folder inside the main dst folder...")
    for img in src.iterdir():
        if not img.is_file():
            continue

        file_name = img.name
        try:
            author, filename = file_name.split("__")
        except ValueError:
            print(f"Skipping (invalid filename): {file_name}")
            skipped += 1
            continue

        target_dir = dst / author
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        if target_path.exists():
            print(f"Skipping (already exists): {target_path}")
            skipped += 1
            continue

        shutil.copy2(str(img), str(target_path))
        print(f"Moved {file_name} to {target_path}")
        moved += 1

    print(f"\nMoved {moved} images.")
    print(f"Skipped {skipped} files.")


def main():
    parser = ArgumentParser(description="Move images from a folder to the correct artist folder based on their name (format: 'author__fname').")
    parser.add_argument("--src", required=True, help="Directory containing the images you want to move.")
    parser.add_argument("--dst", required=True, help="Main output directory with the artist folders.")

    args = parser.parse_args()

    restore_images(args.src, args.dst)


if __name__ == "__main__":
    main()
