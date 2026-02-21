import os
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

#inceptionv3 -> (299x299)
#resnet50 -> (224x224)

TARGET_SIZE = (224, 224)
PADDING_COLOR = (0, 0, 0)
VALID_EXTS = ('.jpg', '.png', '.jpeg')

def resize_and_pad_image(src_path, target_size=TARGET_SIZE, color=PADDING_COLOR):
    try:
        with Image.open(src_path) as image:
            image = image.convert("RGB")

            target_w, target_h = target_size
            w, h = image.size
            scale_factor = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)

            image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            padded_image = Image.new("RGB", target_size, color)
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            padded_image.paste(image_resized, (x,y))

            return padded_image
    except Exception as e:
        print(f"Couldn't process image at path '{src_path}' ({e})")

def process_folder(src_root, dst_root):
    src_root = Path(src_root)

    if not src_root.exists():
        print(f"Source folder {str(src_root)} does not exist.")
        return
    
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"Resizing images in {str(src_root)} to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}px")
    
    for (root, dirs, files) in os.walk(src_root):
        root_path = Path(root)
        for filename in files:
            if not filename.lower().endswith(VALID_EXTS):
                continue
            
            src_path = root_path / filename
            rel_path = root_path.relative_to(src_root)
            dst_path = dst_root / rel_path / filename
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            processed_image = resize_and_pad_image(src_path, TARGET_SIZE, PADDING_COLOR)
            processed_image.save(dst_path, quality=100)

    print("Resizing completed.")

def main():
    parser = ArgumentParser(description=f"Resize and pad images (target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels, padding color: black).")
    parser.add_argument("--src", type=str, required=True, help="Source folder with images organized in artist folders.")
    parser.add_argument("--dst", type=str, required=True, help="Destination folder for processed images.")
    args = parser.parse_args()
    process_folder(args.src,args.dst)

if __name__ == "__main__":
    main()
