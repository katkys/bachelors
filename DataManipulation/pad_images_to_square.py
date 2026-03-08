import os
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

PADDING_COLOR = (0, 0, 0)
VALID_EXTS = ('.jpg', '.png', '.jpeg')

def pad_image_to_square(src_path, color=PADDING_COLOR):
    try:
        with Image.open(src_path) as image:
            image = image.convert("RGB")

            w, h = image.size
            target_size = max(w,h)

            padded_image = Image.new("RGB", (target_size, target_size), color)
            x = (target_size - w) // 2
            y = (target_size - h) // 2
            padded_image.paste(image, (x,y))

            return padded_image
    except Exception as e:
        print(f"Couldn't process image at path '{src_path}' ({e})")

def process_folder(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"Padding images in {str(src_root)} to be square...")
    
    for (root, dirs, files) in os.walk(src_root):
        root_path = Path(root)
        for filename in files:
            if not filename.lower().endswith(VALID_EXTS):
                continue
            
            src_path = root_path / filename
            rel_path = root_path.relative_to(src_root)
            dst_path = dst_root / rel_path / filename
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            processed_image = pad_image_to_square(src_path, PADDING_COLOR)
            processed_image.save(dst_path, quality=100)

    print("Process completed.")

def main():
    parser = ArgumentParser(description=f"Pad images to be square, padding color: black).")
    parser.add_argument("--src", type=str, required=True, help="Source folder with images you wish to process.")
    parser.add_argument("--dst", type=str, required=True, help="Destination folder where processed images will be saved in a mirrored structure.")
    args = parser.parse_args()
    process_folder(args.src,args.dst)

if __name__ == "__main__":
    main()
