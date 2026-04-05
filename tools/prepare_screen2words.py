from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download RICO Screen2Words into a local manifest.")
    parser.add_argument("--dataset", default="rootsautomation/RICO-Screen2Words")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="./datasets/screen2words")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional shuffle seed used before applying --max-images.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Optional fraction of the sampled rows to mark as val inside the output manifest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    if args.val_fraction < 0 or args.val_fraction >= 1:
        raise ValueError("--val-fraction must be in [0, 1).")

    dataset = load_dataset(args.dataset, split=args.split)
    if args.sample_seed is not None and args.max_images is not None:
        dataset = dataset.shuffle(seed=args.sample_seed)

    written = 0
    selected_rows = []
    target_rows = args.max_images or len(dataset)

    for index, row in enumerate(tqdm(dataset, desc="Preparing Screen2Words")):
        captions = [caption.strip() for caption in row.get("captions", []) if isinstance(caption, str) and caption.strip()]
        if row.get("play_store_name"):
            captions.append(str(row["play_store_name"]).strip())
        if row.get("category"):
            captions.append(str(row["category"]).strip())
        captions = sorted(set(captions))
        if not captions:
            continue

        selected_rows.append((index, row, captions))
        if len(selected_rows) >= target_rows:
            break

    train_cutoff = len(selected_rows)
    if args.val_fraction > 0:
        train_cutoff = max(1, int(len(selected_rows) * (1.0 - args.val_fraction)))

    with open(manifest_path, "w", encoding="utf-8") as handle:
        for order_index, (index, row, captions) in enumerate(selected_rows):
            image = row["image"].convert("RGB")
            file_name = row.get("file_name") or f"screen_{index:06d}.jpg"
            image_path = image_dir / file_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, format="JPEG", quality=95)

            if args.val_fraction > 0:
                row_split = "train" if order_index < train_cutoff else "val"
            else:
                row_split = "val" if args.split == "val" else "train" if args.split == "train" else "test"

            payload = {
                "image_path": image_path.as_posix(),
                "captions": captions,
                "split": row_split,
                "source": "screen2words",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"saved_rows={written}")
    print(f"manifest_path={manifest_path}")


if __name__ == "__main__":
    main()
