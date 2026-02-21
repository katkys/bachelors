from argparse import ArgumentParser
import random
import shutil
from pathlib import Path

IMG_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def sample_images(src_dir, dst_dir, num_samples):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    artist_dirs = [d for d in src_dir.iterdir() if d.is_dir()]

    print(f"Found {len(artist_dirs)} artist folders.")

    for artist_dir in artist_dirs:
        artist_name = artist_dir.name
        print(f"\nProcessing artist: {artist_name}")

        images = [img for img in artist_dir.iterdir() if img.suffix.lower() in IMG_FILE_EXTENSIONS]

        if len(images) > 0:
            chosen_samples = random.sample(images, min(num_samples, len(images)))

            output_artist_dir = dst_dir / artist_name
            output_artist_dir.mkdir(parents=True, exist_ok=True)

            for img in chosen_samples:
                destination_path = output_artist_dir / img.name
                shutil.copy2(img, destination_path)
                print(f"  Copied: {img.name}")
        else:      
            print(f"  No images found.")

    print("\nFinished creating image samples.")


def main():
    parser = ArgumentParser(description="Sample a fixed number of images per artist from the source directory to the destination directory.")
    parser.add_argument("--src", type=str, required=True, help="Source directory containing artist subdirectories with images.")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory for sampled images.")
    parser.add_argument("--num", type=int, default=5, help="Number of samples per artist.")
    args = parser.parse_args()
    sample_images(args.src, args.dst, args.num)

if __name__ == "__main__":
    main()
