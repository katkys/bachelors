from argparse import ArgumentParser
from pathlib import Path
import random
import shutil

RANDOM_SEED = 373

IMG_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def sample_images(src_dir, dst_dir, max_samples):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    for artist_dir in src_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        img_files = [f for f in artist_dir.iterdir() if f.suffix.lower() in IMG_FILE_EXTENSIONS]
        
        img_bases = [f.stem.split("_face")[0] for f in img_files]
        
        img_bases_unique = sorted(set(img_bases))
        sample_count = min(len(img_bases_unique), max_samples)
        samples = random.sample(img_bases_unique, sample_count)
        
        dst_artist_dir = dst_dir / artist_dir.name
        dst_artist_dir.mkdir(exist_ok=True)

        for img_file in img_files:
            if img_file.stem.split("_face")[0] in samples:
                shutil.copy2(img_file, dst_artist_dir / img_file.name)


def main():
    parser = ArgumentParser(description="Creates a copy of (unsplit) dataset but for each artist a maximum of 'max' random samples are used.")
    parser.add_argument("--src", type=str, required=True, help="Source directory containing artist subdirectories with images.")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory for sampled images.")
    parser.add_argument("--max", type=int, default=60, help="Maximum number of samples allowed per artist.")
    args = parser.parse_args()

    sample_images(args.src, args.dst, args.max)

if __name__ == "__main__":
    main()
