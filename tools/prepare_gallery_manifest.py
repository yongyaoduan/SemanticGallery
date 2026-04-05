from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"}

DEFAULT_FOLDER_ALIASES = {
    "Screenshots": ["screenshot", "screen capture", "mobile app screenshot"],
    "Camera": ["camera photo", "mobile photo", "daily life photo"],
    "Documents": ["document photo", "paper document", "important document"],
    "Favorites": ["favorite photo", "liked image"],
    "Downloads": ["downloaded image"],
}
def parse_args():
    parser = argparse.ArgumentParser(description="Create a weakly supervised manifest from a private gallery.")
    parser.add_argument("--gallery-path", required=True)
    parser.add_argument("--output-path", default="./datasets/private_gallery/manifest.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--source-name", default="private_gallery_weak")
    return parser.parse_args()


def collect_images(gallery_path: Path):
    return sorted(
        path
        for path in gallery_path.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(gallery_path).parts)
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

def deterministic_split(path: Path, val_ratio: float) -> str:
    digest = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def infer_captions(gallery_path: Path, image_path: Path, folder_aliases: dict[str, list[str]]):
    relative_parts = image_path.relative_to(gallery_path).parts
    captions = set()
    top_level = relative_parts[0] if relative_parts else ""
    stem = image_path.stem.replace("_", " ").replace("-", " ").strip()

    if top_level:
        captions.add(top_level)
        captions.update(folder_aliases.get(top_level, []))

    if stem:
        captions.add(stem)

    path_lower = image_path.as_posix().lower()
    if "screenshot" in path_lower:
        captions.update(["screenshot", "mobile screenshot", "screen capture"])
    if "document" in path_lower or "scan" in path_lower:
        captions.update(["document photo", "scanned document"])

    return sorted(caption for caption in captions if caption and len(caption.strip()) > 1)

def main():
    args = parse_args()
    gallery_path = Path(args.gallery_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in collect_images(gallery_path):
        captions = infer_captions(
            gallery_path=gallery_path,
            image_path=image_path,
            folder_aliases=DEFAULT_FOLDER_ALIASES,
        )
        if not captions:
            continue
        rows.append(
            {
                "image_path": image_path.as_posix(),
                "captions": captions,
                "split": deterministic_split(image_path, args.val_ratio),
                "source": args.source_name,
            }
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved_rows={len(rows)}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
