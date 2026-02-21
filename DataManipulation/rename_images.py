import os
from argparse import ArgumentParser

def rename_images_sequentially(main_dir):
    renamed = 0
    try:
        print("Renaming images...")
        for (root, dirs, files) in os.walk(main_dir):
            images = [f for f in files if f.lower().endswith(('.jpg','.png','.jpeg'))]
            images.sort()
            for i, filename in enumerate(images, start=1):
                renamed += 1
                file_extension = os.path.splitext(filename)[1]
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, f"{i}{file_extension}")
                os.rename(old_path, new_path)
        print(f"Finished renaming {renamed} images.")
    except FileNotFoundError:
        print("Couldn't find the source directory at given path. Try checking the path and rerun the program.")

def main():
    parser = ArgumentParser(description="Rename images sequentially inside a directory and its subdirectories.")
    parser.add_argument("--src", type=str, required=True, help="Path to the main directory which you want to process (can contain images or subdirectories with images).")
    args = parser.parse_args()
    rename_images_sequentially(args.src)


if __name__ == "__main__":
    main()

    
