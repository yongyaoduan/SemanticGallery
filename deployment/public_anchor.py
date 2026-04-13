from __future__ import annotations

import json
from pathlib import Path


def _normalized_screen2words_path(raw_path: str, screen2words_root: Path) -> str | None:
    images_root = screen2words_root / "images"
    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        resolved = (screen2words_root / path).resolve()
        if resolved.is_file():
            return path.as_posix()

    if path.is_absolute() and path.is_file():
        try:
            return path.resolve().relative_to(screen2words_root.resolve()).as_posix()
        except ValueError:
            pass

    parts = [part for part in path.parts if part not in {"", "/"}]
    if "images" not in parts:
        return None

    suffix = parts[parts.index("images") + 1 :]
    if not suffix:
        return None

    candidate = images_root.joinpath(*suffix)
    if not candidate.is_file():
        return None
    return Path("images").joinpath(*suffix).as_posix()


def normalize_public_anchor_extract(extract_root: str | Path) -> int:
    extract_root = Path(extract_root).expanduser().resolve()
    manifest_path = extract_root / "screen2words" / "manifest.jsonl"
    screen2words_root = manifest_path.parent

    if not manifest_path.is_file():
        return 0

    rows = []
    repaired_rows = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        normalized_path = _normalized_screen2words_path(str(row.get("image_path", "")), screen2words_root)
        if normalized_path and row.get("image_path") != normalized_path:
            row["image_path"] = normalized_path
            repaired_rows += 1
        rows.append(row)

    if repaired_rows:
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return repaired_rows
