from __future__ import annotations

import hashlib
from pathlib import Path

SUPPORTED_GALLERY_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"}


def iter_gallery_paths(gallery_path: str | Path) -> list[Path]:
    root = Path(gallery_path).expanduser().resolve()
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.suffix.lower() in SUPPORTED_GALLERY_SUFFIXES
    )


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_gallery_bank_state_payload(
    gallery_path: str | Path,
    *,
    model_dir: str | Path,
    precision: str,
    weights_file_path: str | Path | None,
) -> dict[str, str | int | None]:
    root = Path(gallery_path).expanduser().resolve()
    payload: dict[str, str | int | None] = {
        "gallery_dir": root.as_posix(),
        "model_dir": Path(model_dir).expanduser().resolve().as_posix(),
        "precision": precision,
    }

    if weights_file_path:
        weights_path = Path(weights_file_path).expanduser().resolve()
        payload["weights_file_path"] = weights_path.as_posix()
        payload["weights_file_sha256"] = sha256_file(weights_path)
    else:
        payload["weights_file_path"] = ""
        payload["weights_file_sha256"] = None

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for image_path in iter_gallery_paths(root):
        stat = image_path.stat()
        relative_path = image_path.relative_to(root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += stat.st_size

    payload["gallery_file_count"] = file_count
    payload["gallery_total_bytes"] = total_bytes
    payload["gallery_state_sha256"] = digest.hexdigest()
    return payload
