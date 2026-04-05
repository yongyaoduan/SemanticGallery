from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a balanced private adaptation manifest capped at a fixed sample count.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260403)
    return parser.parse_args()


def load_manifest(path: str) -> list[dict]:
    rows = []
    manifest_path = Path(path).expanduser().resolve()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {manifest_path}")
    return rows


def infer_folder_name(row: dict) -> str:
    image_path = Path(row["image_path"]).expanduser()
    if len(image_path.parts) >= 2:
        return image_path.parent.name
    return "unknown"


def balanced_sample(rows: list[dict], target_size: int, seed: int):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[infer_folder_name(row)].append(row)

    for folder_rows in grouped.values():
        rng.shuffle(folder_rows)

    selected = []
    selected_ids = set()

    def take(folder: str, limit: int):
        taken = 0
        for row in grouped.get(folder, []):
            row_id = row["image_path"]
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            taken += 1
            if taken >= limit or len(selected) >= target_size:
                break

    folder_order = sorted(
        grouped,
        key=lambda folder: (
            -len(grouped[folder]),
            folder,
        ),
    )

    while len(selected) < target_size:
        progress = False
        for folder in folder_order:
            before = len(selected)
            take(folder, 1)
            if len(selected) > before:
                progress = True
            if len(selected) >= target_size:
                break
        if not progress:
            break

    return selected


def main():
    args = parse_args()
    if args.target_size > 100:
        raise ValueError("Private adaptation manifests must stay at or below 100 rows.")

    rows = load_manifest(args.source_manifest)
    effective_target_size = min(args.target_size, len(rows))
    selected = balanced_sample(
        rows=rows,
        target_size=effective_target_size,
        seed=args.seed,
    )

    if len(selected) < effective_target_size:
        raise ValueError(f"Only sampled {len(selected)} rows, below requested target_size={effective_target_size}")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(infer_folder_name(row) for row in selected)
    print(f"saved_rows={len(selected)}")
    print(f"target_size={effective_target_size}")
    print(f"output_path={output_path}")
    print(json.dumps({"folder_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
