from __future__ import annotations

import hashlib
from pathlib import Path

from deployment.gallery_state import SUPPORTED_GALLERY_SUFFIXES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_visible_images(folder_path: Path) -> list[Path]:
    root = folder_path.expanduser().resolve()
    return sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_GALLERY_SUFFIXES
        and not any(part.startswith(".") for part in candidate.relative_to(root).parts)
    )


def build_scan_signature(rows: list[tuple[str, int, int]]) -> str:
    digest = hashlib.sha256()
    for relative_path, size, mtime_ns in rows:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(mtime_ns).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def mark_missing_paths(store, folder_path: Path, seen_paths: set[str]) -> None:
    folder_root = folder_path.expanduser().resolve().as_posix()
    for row in store.get_known_paths(folder_root):
        if row["absolute_path"] not in seen_paths and row["is_present"]:
            store.mark_path_missing(row["absolute_path"])


def reconcile_folder(store, folder_path: Path, encoder_signature: str, encoder) -> None:
    folder_root = folder_path.expanduser().resolve()
    existing_rows = {
        row["absolute_path"]: row
        for row in store.get_known_paths(folder_root.as_posix())
    }
    seen_paths: set[str] = set()
    signature_rows: list[tuple[str, int, int]] = []

    for path in iter_visible_images(folder_root):
        stat = path.stat()
        absolute_path = path.resolve().as_posix()
        seen_paths.add(absolute_path)
        signature_rows.append((path.relative_to(folder_root).as_posix(), stat.st_size, stat.st_mtime_ns))

        existing_row = existing_rows.get(absolute_path)
        unchanged = (
            existing_row is not None
            and existing_row["byte_size"] == stat.st_size
            and existing_row["mtime_ns"] == stat.st_mtime_ns
        )

        if unchanged:
            content_hash = existing_row["content_hash"]
        else:
            content_hash = sha256_file(path)
            store.upsert_asset(content_hash=content_hash, byte_size=stat.st_size)
            if store.get_embedding_row(content_hash, encoder_signature) is None:
                store.upsert_embedding(content_hash, encoder_signature, encoder.encode_image(path))

        store.upsert_path(
            absolute_path=absolute_path,
            folder_path=folder_root.as_posix(),
            content_hash=content_hash,
            byte_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            is_present=True,
        )

    mark_missing_paths(store, folder_root, seen_paths)
    scan_signature = build_scan_signature(signature_rows)
    store.upsert_folder_state(
        folder_root.as_posix(),
        encoder_signature,
        file_count=len(signature_rows),
        total_bytes=sum(size for _, size, _ in signature_rows),
        scan_signature=scan_signature,
    )
