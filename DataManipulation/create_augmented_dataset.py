import os
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

VALID_EXTS = ('.jpg', '.png', '.jpeg')

def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rotate_left(image):
    return image.rotate(12,resample=Image.BICUBIC)

def rotate_right(image):
    return image.rotate(-12,resample=Image.BICUBIC)

def zoom_in(image):
    w, h = image.size

    scale = 1.3
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_size=(new_w, new_h)

    resized = image.resize(new_size, resample=Image.BICUBIC)

    left = (new_w - w) // 2
    top = (new_h - h) // 2

    return resized.crop((left, top, left + w, top + h))


AUGMENTATIONS = [
    ("aug1", rotate_left),
    ("aug2", rotate_right),
    ("aug3", zoom_in),
    ("aug4", horizontal_flip)
]


def process_folder(src_root, dst_root):
    src_root = Path(src_root)

    if not src_root.exists():
        print(f"Source folder '{str(src_root)}' does not exist.")
        return

    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True)

    print(f"Creating deterministic augmented dataset from '{str(src_root)}'...")

    images_count = 0
    for (root, dirs, files) in os.walk(src_root):
        root_path = Path(root)
        for filename in files:
            if not filename.lower().endswith(VALID_EXTS):
                continue

            src_path = root_path / filename
            rel_path = root_path.relative_to(src_root)

            dst_dir = dst_root / rel_path
            dst_dir.mkdir(parents=True, exist_ok=True)

            is_test = "test" in rel_path.parts

            try:
                with Image.open(src_path) as image:
                    image = image.convert("RGB")
                    stem = src_path.stem
                    suffix = src_path.suffix

                    # save copy of original image
                    original_dst = dst_dir / f"{stem}{suffix}"
                    image.save(original_dst, quality=100)

                    # create and save augmented versions (except for test samples)
                    if not is_test:  
                        for aug_name, aug_fn in AUGMENTATIONS:
                            augmented = aug_fn(image)

                            aug_fname = (f"{stem}_{aug_name}{suffix}")
                            aug_dst = dst_dir / aug_fname
                            augmented.save(aug_dst, quality=100)

                    images_count += 1

            except Exception as e:
                print(f"Couldn't process image '{src_path}' ({e})")

    print("\nAugmentation completed.")
    print(f"Augmented dataset saved to: '{str(dst_root)}'")


def main():
    parser = ArgumentParser(description=("Create deterministic offline augmented dataset with mirrored folder structure."))
    parser.add_argument("--src", type=str, required=True, help="Path to the main source directory.")
    parser.add_argument("--dst", type=str, required=True, help="Path to the destination directory.")
    args = parser.parse_args()

    process_folder(args.src, args.dst)


if __name__ == "__main__":
    main()