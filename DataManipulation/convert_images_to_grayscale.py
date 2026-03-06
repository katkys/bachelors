import os
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

VALID_EXTS = ('.jpg', '.png', '.jpeg')

def convert_to_grayscale(src_path):
    try:
        with Image.open(src_path) as image:
            gray_image = image.convert("L")  
            gray_image = gray_image.convert("RGB") 
            return gray_image
    except Exception as e:
        print(f"Couldn't process image at path '{src_path}' ({e})")
        return None

def process_folder(src_root, dst_root):
    src_root = Path(src_root)

    if not src_root.exists():
        print(f"Source folder {str(src_root)} does not exist.")
        return
    
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"Converting images in {str(src_root)} to grayscale...")
    
    for (root, dirs, files) in os.walk(src_root):
        root_path = Path(root)
        for filename in files:
            if not filename.lower().endswith(VALID_EXTS):
                continue
            
            src_path = root_path / filename
            rel_path = root_path.relative_to(src_root)
            dst_path = dst_root / rel_path / filename
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            processed_image = convert_to_grayscale(src_path)
            processed_image.save(dst_path, quality=100)
    
    print("Conversion completed. Grayscale images saved to: ", str(dst_root))

def main():
    parser = ArgumentParser(description="Save grayscale versions of original images in a mirrored folder structure.")
    parser.add_argument("--src", type=str, required=True, help="Source folder with images organized in artist folders.")
    parser.add_argument("--dst", type=str, required=True, help="Main destination folder for grayscale images.")
    args = parser.parse_args()
    process_folder(args.src, args.dst)

if __name__ == "__main__":
    main()
