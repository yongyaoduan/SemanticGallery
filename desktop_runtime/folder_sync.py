from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


@dataclass(frozen=True)
class FolderScanEntry:
    path: Path
    relative_path: str
    byte_size: int
    mtime_ns: int


def collect_folder_scan_entries(folder_path: Path) -> list[FolderScanEntry]:
    root = folder_path.expanduser().resolve()
    entries: list[FolderScanEntry] = []
    for path in iter_visible_images(root):
        stat = path.stat()
        entries.append(
            FolderScanEntry(
                path=path.resolve(),
                relative_path=path.relative_to(root).as_posix(),
                byte_size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return entries


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


def reconcile_folder(
    store,
    folder_path: Path,
    encoder_signature: str,
    encoder,
    *,
    scan_entries: list[FolderScanEntry] | None = None,
    emit_progress: Callable[[dict[str, object]], None] | None = None,
) -> None:
    folder_root = folder_path.expanduser().resolve()
    entries = scan_entries or collect_folder_scan_entries(folder_root)
    existing_rows = {
        row["absolute_path"]: row
        for row in store.get_known_paths(folder_root.as_posix())
    }
    seen_paths: set[str] = set()
    signature_rows: list[tuple[str, int, int]] = []
    pending_entries: list[tuple[FolderScanEntry, object | None]] = []
    embedded_count = 0
    reused_count = 0
    total_entries = len(entries)

    for entry in entries:
        absolute_path = entry.path.as_posix()
        seen_paths.add(absolute_path)
        signature_rows.append((entry.relative_path, entry.byte_size, entry.mtime_ns))

        existing_row = existing_rows.get(absolute_path)
        unchanged = (
            existing_row is not None
            and existing_row["is_present"]
            and existing_row["byte_size"] == entry.byte_size
            and existing_row["mtime_ns"] == entry.mtime_ns
        )
        if unchanged and store.get_embedding_row(existing_row["content_hash"], encoder_signature) is not None:
            continue

        pending_entries.append((entry, existing_row))

    if emit_progress is not None:
        emit_progress(
            {
                "phase": "start",
                "current": 0,
                "total": total_entries,
                "embeddedCount": embedded_count,
                "reusedCount": reused_count,
                "message": "Scanning the selected folder for index updates.",
            }
        )

    ready_count = total_entries - len(pending_entries)
    for index, (entry, existing_row) in enumerate(pending_entries, start=1):
        absolute_path = entry.path.as_posix()
        unchanged = (
            existing_row is not None
            and existing_row["is_present"]
            and existing_row["byte_size"] == entry.byte_size
            and existing_row["mtime_ns"] == entry.mtime_ns
        )

        if unchanged:
            content_hash = existing_row["content_hash"]
            if store.get_embedding_row(content_hash, encoder_signature) is None:
                store.upsert_embedding(content_hash, encoder_signature, encoder.encode_image(entry.path))
                embedded_count += 1
            else:
                reused_count += 1
        else:
            content_hash = sha256_file(entry.path)
            store.upsert_asset(content_hash=content_hash, byte_size=entry.byte_size)
            if store.get_embedding_row(content_hash, encoder_signature) is None:
                store.upsert_embedding(content_hash, encoder_signature, encoder.encode_image(entry.path))
                embedded_count += 1
            else:
                reused_count += 1

        store.upsert_path(
            absolute_path=absolute_path,
            folder_path=folder_root.as_posix(),
            content_hash=content_hash,
            byte_size=entry.byte_size,
            mtime_ns=entry.mtime_ns,
            is_present=True,
        )

        if emit_progress is not None:
            processed_count = ready_count + index
            emit_progress(
                {
                    "phase": "progress",
                    "current": processed_count,
                    "total": total_entries,
                    "fileName": entry.path.name,
                    "embeddedCount": embedded_count,
                    "reusedCount": reused_count,
                    "message": f"Indexing {entry.path.name} ({processed_count}/{total_entries})",
                }
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

    if emit_progress is not None:
        emit_progress(
            {
                "phase": "finish",
                "current": total_entries,
                "total": total_entries,
                "embeddedCount": embedded_count,
                "reusedCount": reused_count,
                "message": "The folder index is ready.",
            }
        )
