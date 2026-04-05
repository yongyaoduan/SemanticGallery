from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download and normalize Flickr30k into the local folder layout.")
    parser.add_argument("--dataset", default="lmms-lab/flickr30k", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="test", help="Dataset split to download.")
    parser.add_argument("--output-dir", default="./datasets/flickr30k", help="Local output directory.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional shuffle seed used before applying --max-images.")
    return parser.parse_args()


def resolve_image(row):
    for key in ("image", "images"):
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, Image.Image):
                return value.convert("RGB")
    raise ValueError(f"Unable to find image field in row keys: {list(row.keys())}")


def resolve_captions(row):
    for key in ("caption", "captions", "text", "sentences"):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
    raise ValueError(f"Unable to find caption field in row keys: {list(row.keys())}")


def resolve_filename(row, index: int) -> str:
    for key in ("filename", "file_name", "image_id", "img_id", "id"):
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                if "." not in value:
                    value = f"{value}.jpg"
                return value
    return f"img_{index:06d}.jpg"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = output_dir / "flickr30k_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    if args.sample_seed is not None and args.max_images is not None:
        dataset = dataset.shuffle(seed=args.sample_seed)
    captions_by_name = defaultdict(list)

    seen_images = set()
    written = 0
    for index, row in enumerate(tqdm(dataset, desc="Preparing Flickr30k")):
        filename = resolve_filename(row, index)
        captions = resolve_captions(row)
        captions_by_name[filename].extend(captions)

        if filename not in seen_images:
            image = resolve_image(row)
            image.save(image_dir / filename, format="JPEG", quality=95)
            seen_images.add(filename)
            written += 1

        if args.max_images is not None and written >= args.max_images:
            break

    captions_path = output_dir / "captions.txt"
    with open(captions_path, "w", encoding="utf-8") as handle:
        handle.write("image_name,caption_idx,caption\n")
        for filename, captions in sorted(captions_by_name.items()):
            for caption_idx, caption in enumerate(captions):
                clean_caption = caption.replace("\n", " ").replace("\r", " ").strip()
                handle.write(f"{filename},{caption_idx},{clean_caption}\n")

    print(f"Saved {len(seen_images)} images to {image_dir}")
    print(f"Saved captions file to {captions_path}")


if __name__ == "__main__":
    main()
