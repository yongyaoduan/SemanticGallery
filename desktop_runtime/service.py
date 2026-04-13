from __future__ import annotations

import asyncio
import hashlib
import io
import json
import mimetypes
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from urllib.parse import quote, unquote

import numpy as np
from PIL import Image, ImageOps

from deployment.search_utils import is_searchable_query
from desktop_runtime.folder_sync import (
    FolderScanEntry,
    build_scan_signature,
    collect_folder_scan_entries,
    iter_visible_images,
    reconcile_folder,
)
from desktop_runtime.search_view import ActiveSearchView
from desktop_runtime.stage2_jobs import Stage2Error
from desktop_runtime.trash import move_paths_to_trash

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional for JPEG/PNG-only galleries
    register_heif_opener = None
else:  # pragma: no cover - covered by integration on machines with pillow-heif
    register_heif_opener()


THUMBNAIL_SIZE = (512, 512)
HEIF_SUFFIXES = {".heic", ".heif"}
MAX_IMAGE_UPLOAD_BYTES = 20 * 1024 * 1024


class DesktopServiceError(RuntimeError):
    pass


class EncoderProtocol(Protocol):
    def encode_image(self, path: Path) -> np.ndarray: ...

    def encode_text(self, query_text: str) -> np.ndarray: ...


class EncoderLoaderProtocol(Protocol):
    def __call__(self, folder_path: Path | None, encoder_signature: str) -> EncoderProtocol: ...


class Stage2JobProtocol(Protocol):
    def run(self, folder_path: Path) -> str: ...


class TrashManagerProtocol(Protocol):
    def __call__(self, paths: list[Path]) -> None: ...


@dataclass(frozen=True)
class FolderScanState:
    file_count: int
    total_bytes: int
    scan_signature: str
    rows: tuple[tuple[Path, str, int, int], ...]


@dataclass(frozen=True)
class ServedImage:
    media_type: str
    file_path: Path | None = None
    content: bytes | None = None


@dataclass
class DesktopService:
    store: Any | None = None
    encoder: EncoderProtocol | None = None
    encoder_loader: EncoderLoaderProtocol | None = None
    stage2_job: Stage2JobProtocol | None = None
    thumbnails_dir: Path | None = None
    trash_manager: TrashManagerProtocol | None = None
    setup_status: str = "idle"
    active_folder: Path | None = None
    active_encoder_signature: str = "stage1"
    last_task_message: str = ""
    watch_interval_seconds: float = 5.0
    _event_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue, init=False)
    _watch_task: asyncio.Task[None] | None = field(default=None, init=False)
    _active_scan_signature: str = field(default="", init=False)
    _loaded_encoder_key: tuple[str | None, str] | None = field(default=None, init=False)
    _index_progress: dict[str, object] = field(
        default_factory=lambda: {
            "status": "idle",
            "phase": "idle",
            "current": 0,
            "total": 0,
            "startedAtMs": None,
            "elapsedSeconds": 0,
            "remainingSeconds": None,
            "embeddedCount": 0,
            "reusedCount": 0,
            "message": "No indexing task is running.",
        },
        init=False,
    )
    _stage2_progress: dict[str, object] = field(
        default_factory=lambda: {
            "status": "idle",
            "phase": "idle",
            "current": 0,
            "total": 0,
            "startedAtMs": None,
            "elapsedSeconds": 0,
            "remainingSeconds": None,
            "message": "Stage 2 adaptation is idle.",
        },
        init=False,
    )
    _index_started_at_monotonic: float | None = field(default=None, init=False, repr=False)
    _index_started_at_ms: int | None = field(default=None, init=False, repr=False)
    _stage2_started_at_monotonic: float | None = field(default=None, init=False, repr=False)
    _stage2_started_at_ms: int | None = field(default=None, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _active_search_view: ActiveSearchView = field(
        default_factory=lambda: ActiveSearchView(paths=[], matrix=np.zeros((0, 0), dtype=np.float32)),
        init=False,
    )
    _event_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _metadata_cache: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    _mutation_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def runtime_status(self) -> dict[str, object]:
        with self._lock:
            return {
                "setupStatus": self.setup_status,
                "activeFolder": self.active_folder.as_posix() if self.active_folder else None,
                "activeEncoderSignature": self.active_encoder_signature,
                "lastTaskMessage": self.last_task_message,
                "indexedImageCount": len(self._active_search_view.paths),
                "indexing": dict(self._index_progress),
                "stage2": dict(self._stage2_progress),
            }

    def attach_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._event_loop = loop

    def set_active_folder(self, folder_path: str) -> dict[str, object]:
        with self._mutation_lock:
            resolved = Path(folder_path).expanduser().resolve()
            if not resolved.is_dir():
                raise DesktopServiceError("The selected folder is not available.")

            with self._lock:
                previous_folder = self.active_folder
                previous_signature = self.active_encoder_signature
                previous_scan_signature = self._active_scan_signature
                previous_encoder = self.encoder
                previous_encoder_key = self._loaded_encoder_key
                previous_view = self._active_search_view
                previous_stage2_progress = dict(self._stage2_progress)
                previous_stage2_started_at_monotonic = self._stage2_started_at_monotonic
                previous_stage2_started_at_ms = self._stage2_started_at_ms

                self.active_folder = resolved
                self.active_encoder_signature = self._resolve_folder_signature(resolved)
                self._reset_stage2_progress_for_signature(self.active_encoder_signature)

            try:
                return self.refresh_active_folder(lightweight=False)
            except Stage2Error:
                with self._lock:
                    self.active_folder = previous_folder
                    self.active_encoder_signature = previous_signature
                    self._active_scan_signature = previous_scan_signature
                    self.encoder = previous_encoder
                    self._loaded_encoder_key = previous_encoder_key
                    self._active_search_view = previous_view
                    self._stage2_progress = previous_stage2_progress
                    self._stage2_started_at_monotonic = previous_stage2_started_at_monotonic
                    self._stage2_started_at_ms = previous_stage2_started_at_ms
                raise
            except DesktopServiceError as exc:
                with self._lock:
                    self.active_folder = previous_folder
                    self.active_encoder_signature = previous_signature
                    self._active_scan_signature = previous_scan_signature
                    self.encoder = previous_encoder
                    self._loaded_encoder_key = previous_encoder_key
                    self._active_search_view = previous_view
                    self._stage2_progress = previous_stage2_progress
                    self._stage2_started_at_monotonic = previous_stage2_started_at_monotonic
                    self._stage2_started_at_ms = previous_stage2_started_at_ms
                    self.last_task_message = str(exc)
                raise DesktopServiceError(f"Failed to index the selected folder: {exc}") from exc
            except Exception as exc:
                with self._lock:
                    self.active_folder = previous_folder
                    self.active_encoder_signature = previous_signature
                    self._active_scan_signature = previous_scan_signature
                    self.encoder = previous_encoder
                    self._loaded_encoder_key = previous_encoder_key
                    self._active_search_view = previous_view
                    self._stage2_progress = previous_stage2_progress
                    self._stage2_started_at_monotonic = previous_stage2_started_at_monotonic
                    self._stage2_started_at_ms = previous_stage2_started_at_ms
                    self.last_task_message = str(exc)
                raise DesktopServiceError(f"Failed to index the selected folder: {exc}") from exc

    def refresh_active_folder(self, lightweight: bool = False) -> dict[str, object]:
        with self._mutation_lock:
            with self._lock:
                folder = self._require_active_folder()
                active_encoder_signature = self.active_encoder_signature
            scan_state = self._scan_folder(folder)

            with self._lock:
                if lightweight and scan_state.scan_signature == self._active_scan_signature:
                    self.last_task_message = "The active folder is already up to date."
                    return {"refreshed": False, "skipped": True, **self.runtime_status()}

                self._require_indexing_components()
                self._publish_index_progress(
                    {
                        "status": "running",
                        "phase": "start",
                        "current": 0,
                        "total": scan_state.file_count,
                        "startedAtMs": None,
                        "elapsedSeconds": 0,
                        "remainingSeconds": None,
                        "embeddedCount": 0,
                        "reusedCount": 0,
                        "folderPath": folder.as_posix(),
                        "message": "Scanning the selected folder for index updates.",
                    }
                )

            try:
                with self._lock:
                    encoder = self._load_encoder(folder, active_encoder_signature)
                reconcile_folder(
                    self.store,
                    folder,
                    active_encoder_signature,
                    encoder,
                    scan_entries=[
                        FolderScanEntry(
                            path=path,
                            relative_path=relative_path,
                            byte_size=byte_size,
                            mtime_ns=mtime_ns,
                        )
                        for path, relative_path, byte_size, mtime_ns in scan_state.rows
                    ],
                    emit_progress=lambda payload: self._publish_index_progress(
                        {
                            "status": "ready" if payload["phase"] == "finish" else "running",
                            "folderPath": folder.as_posix(),
                            **payload,
                        }
                    ),
                )
                next_view = ActiveSearchView.from_store(self.store, folder.as_posix(), active_encoder_signature)
            except (DesktopServiceError, Stage2Error):
                raise
            except Exception as exc:
                with self._lock:
                    self.last_task_message = str(exc)
                    self._publish_index_progress(
                        {
                            "status": "failed",
                            "phase": "failed",
                            "current": 0,
                            "total": scan_state.file_count,
                            "startedAtMs": None,
                            "elapsedSeconds": 0,
                            "remainingSeconds": None,
                            "embeddedCount": 0,
                            "reusedCount": 0,
                            "folderPath": folder.as_posix(),
                            "message": f"Failed to refresh the active folder: {exc}",
                        }
                    )
                raise DesktopServiceError(f"Failed to refresh the active folder: {exc}") from exc

            with self._lock:
                self._active_search_view = next_view
                self._active_scan_signature = scan_state.scan_signature
                self.last_task_message = "The active folder index is ready."
            self.publish_event(
                "folder-refreshed",
                {
                    "folderPath": folder.as_posix(),
                    "indexedImageCount": len(next_view.paths),
                    "lightweight": lightweight,
                },
            )
            return {"refreshed": True, "skipped": False, **self.runtime_status()}

    def run_stage2_for_active_folder(self) -> dict[str, object]:
        with self._mutation_lock:
            with self._lock:
                folder = self._require_active_folder()
                if self.stage2_job is None:
                    raise DesktopServiceError("Stage 2 adaptation is not available.")
                self._require_indexing_components()
                previous_encoder = self.encoder
                previous_encoder_key = self._loaded_encoder_key
                self._publish_stage2_progress(
                    {
                        "status": "running",
                        "phase": "validate",
                        "current": 0,
                        "total": 0,
                        "message": "Checking the active folder for Stage 2 adaptation.",
                    }
                )

            try:
                next_signature = self.stage2_job.run(folder)
                with self._lock:
                    next_encoder = self._load_encoder(folder, next_signature)
                reconcile_folder(self.store, folder, next_signature, next_encoder)
                next_view = ActiveSearchView.from_store(self.store, folder.as_posix(), next_signature)
            except (DesktopServiceError, Stage2Error) as exc:
                with self._lock:
                    self.encoder = previous_encoder
                    self._loaded_encoder_key = previous_encoder_key
                    self.last_task_message = str(exc)
                    current = int(self._stage2_progress.get("current", 0) or 0)
                    total = int(self._stage2_progress.get("total", 0) or 0)
                    self._publish_stage2_progress(
                        {
                            "status": "failed",
                            "phase": "failed",
                            "current": current,
                            "total": max(total, current),
                            "message": str(exc),
                        }
                    )
                raise
            except Exception as exc:
                with self._lock:
                    self.encoder = previous_encoder
                    self._loaded_encoder_key = previous_encoder_key
                    self.last_task_message = str(exc)
                    current = int(self._stage2_progress.get("current", 0) or 0)
                    total = int(self._stage2_progress.get("total", 0) or 0)
                    self._publish_stage2_progress(
                        {
                            "status": "failed",
                            "phase": "failed",
                            "current": current,
                            "total": max(total, current),
                            "message": str(exc),
                        }
                    )
                raise DesktopServiceError(f"Failed to finish Stage 2 adaptation: {exc}") from exc
            else:
                with self._lock:
                    self.encoder = next_encoder
                    self._loaded_encoder_key = self._encoder_cache_key(folder, next_signature)

            with self._lock:
                self.active_encoder_signature = next_signature
                self._active_search_view = next_view
                self._active_scan_signature = self._scan_folder(folder).scan_signature
                self.last_task_message = "Stage 2 adaptation is ready."
                completed_total = int(self._stage2_progress.get("total", 0) or 0)
                if completed_total <= 0:
                    completed_total = max(1, int(self._stage2_progress.get("current", 0) or 0), 1)
                self._publish_stage2_progress(
                    {
                        "status": "ready",
                        "phase": "finish",
                        "current": completed_total,
                        "total": completed_total,
                        "message": "Stage 2 adaptation is ready.",
                    }
                )
            self.publish_event(
                "stage2-complete",
                {
                    "folderPath": folder.as_posix(),
                    "activeEncoderSignature": next_signature,
                    "indexedImageCount": len(next_view.paths),
                },
            )
            return {"stage2Ran": True, **self.runtime_status()}

    def search_text(self, query_text: str, limit: int) -> dict[str, object]:
        with self._lock:
            text = query_text.strip()
            if not is_searchable_query(text) or limit <= 0 or not self._active_search_view.paths:
                return {"query": text, "results": []}
            folder = self._require_active_folder()
            encoder = self._load_encoder(folder, self.active_encoder_signature)
            try:
                query_vector = encoder.encode_text(text)
            except Exception as exc:
                raise DesktopServiceError(f"Failed to encode the search query: {exc}") from exc
            matches = self._active_search_view.search(query_vector, limit)
            return {"query": text, "results": [self._result_payload(path) for path in matches]}

    def search_image(self, file_bytes: bytes, limit: int, filename: str | None = None) -> dict[str, object]:
        with self._lock:
            if limit <= 0 or not self._active_search_view.paths:
                return {"query": "", "results": []}

            query_vector = self._encode_uploaded_image(file_bytes, filename)
            matches = self._active_search_view.search(query_vector, limit)
            return {"query": "", "results": [self._result_payload(path) for path in matches]}

    def search_similar(self, image_path: str, limit: int) -> dict[str, object]:
        with self._lock:
            file_path = self._resolve_active_file(image_path)
            if limit <= 0 or not self._active_search_view.paths:
                return {"query": "", "results": []}

            matches = self._active_search_view.search_similar(file_path.as_posix(), limit)
            if not matches:
                encoder = self._load_encoder(self._require_active_folder(), self.active_encoder_signature)
                try:
                    query_vector = encoder.encode_image(file_path)
                except Exception as exc:
                    raise DesktopServiceError(f"Failed to encode the reference image: {exc}") from exc
                matches = self._active_search_view.search(query_vector, limit, exclude_path=file_path.as_posix())

            return {"query": "", "results": [self._result_payload(path) for path in matches]}

    def metadata(self, image_path: str) -> dict[str, object]:
        with self._lock:
            file_path = self._resolve_active_file(image_path)
            return dict(self._read_image_metadata(file_path))

    def thumbnail_path(self, image_path: str) -> Path:
        with self._lock:
            file_path = self._resolve_active_file(image_path)
            return self._ensure_thumbnail(file_path)

    def image_asset(self, image_path: str) -> ServedImage:
        with self._lock:
            file_path = self._resolve_active_file(image_path)
            return self._serve_original_image(file_path)

    def delete_image(self, image_path: str) -> dict[str, object]:
        with self._mutation_lock:
            with self._lock:
                payload = self._trash_active_files([image_path])
                if not payload["deleted"]:
                    raise DesktopServiceError("Image not found.")
                deleted = payload["deleted"][0]
                return {
                    "deleted": True,
                    "fileName": deleted["fileName"],
                    "path": deleted["path"],
                    "relativePath": deleted["relativePath"],
                    "message": payload["message"],
                }

    def delete_images(self, image_paths: list[str]) -> dict[str, object]:
        with self._mutation_lock:
            with self._lock:
                return self._trash_active_files(image_paths)

    def publish_event(self, event_name: str, payload: dict[str, object]) -> None:
        event = {"event": event_name, "payload": payload}
        if self._event_loop is None or self._event_loop.is_closed():
            self._event_queue.put_nowait(event)
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._event_loop:
            self._event_queue.put_nowait(event)
            return

        self._event_loop.call_soon_threadsafe(self._event_queue.put_nowait, event)

    def _publish_index_progress(self, payload: dict[str, object]) -> None:
        with self._lock:
            status = str(payload.get("status", self._index_progress.get("status", "idle")))
            phase = str(payload.get("phase", "idle"))
            next_payload = dict(payload)

            if status == "running":
                if phase == "start" or self._index_started_at_monotonic is None or self._index_started_at_ms is None:
                    self._index_started_at_monotonic = time.monotonic()
                    self._index_started_at_ms = int(time.time() * 1000)
                elapsed_seconds = max(0, int(round(time.monotonic() - self._index_started_at_monotonic)))
                current = int(next_payload.get("current", 0) or 0)
                total = int(next_payload.get("total", 0) or 0)
                remaining_seconds = None
                if total > 0 and current > 0 and total >= current:
                    remaining_seconds = int(round((elapsed_seconds / current) * (total - current)))
                next_payload.update(
                    {
                        "startedAtMs": self._index_started_at_ms,
                        "elapsedSeconds": elapsed_seconds,
                        "remainingSeconds": remaining_seconds,
                    }
                )
            elif status == "ready":
                elapsed_seconds = 0
                if self._index_started_at_monotonic is not None:
                    elapsed_seconds = max(0, int(round(time.monotonic() - self._index_started_at_monotonic)))
                next_payload.update(
                    {
                        "startedAtMs": self._index_started_at_ms,
                        "elapsedSeconds": elapsed_seconds,
                        "remainingSeconds": 0,
                    }
                )
                self._index_started_at_monotonic = None
                self._index_started_at_ms = None
            else:
                next_payload.update(
                    {
                        "startedAtMs": None,
                        "elapsedSeconds": 0,
                        "remainingSeconds": None,
                    }
                )
                self._index_started_at_monotonic = None
                self._index_started_at_ms = None

            self._index_progress = next_payload
            self.last_task_message = str(next_payload.get("message", self.last_task_message))
        self.publish_event("folder-index-progress", next_payload)

    def _publish_stage2_progress(self, payload: dict[str, object]) -> None:
        with self._lock:
            status = str(payload.get("status", self._stage2_progress.get("status", "idle")))
            phase = str(payload.get("phase", "idle"))
            next_payload = dict(payload)

            if status == "running":
                if phase == "prepare" or self._stage2_started_at_monotonic is None or self._stage2_started_at_ms is None:
                    self._stage2_started_at_monotonic = time.monotonic()
                    self._stage2_started_at_ms = int(time.time() * 1000)
                elapsed_seconds = max(0, int(round(time.monotonic() - self._stage2_started_at_monotonic)))
                current = int(next_payload.get("current", 0) or 0)
                total = int(next_payload.get("total", 0) or 0)
                remaining_seconds = next_payload.get("remainingSeconds")
                if remaining_seconds is None:
                    if total > 0 and current > 0 and total >= current:
                        remaining_seconds = int(round((elapsed_seconds / current) * (total - current)))
                else:
                    remaining_seconds = int(remaining_seconds)
                next_payload.update(
                    {
                        "startedAtMs": self._stage2_started_at_ms,
                        "elapsedSeconds": elapsed_seconds,
                        "remainingSeconds": remaining_seconds,
                    }
                )
            elif status == "ready":
                elapsed_seconds = 0
                if self._stage2_started_at_monotonic is not None:
                    elapsed_seconds = max(0, int(round(time.monotonic() - self._stage2_started_at_monotonic)))
                next_payload.update(
                    {
                        "startedAtMs": self._stage2_started_at_ms,
                        "elapsedSeconds": elapsed_seconds,
                        "remainingSeconds": 0,
                    }
                )
                self._stage2_started_at_monotonic = None
                self._stage2_started_at_ms = None
            else:
                next_payload.update(
                    {
                        "startedAtMs": None,
                        "elapsedSeconds": 0,
                        "remainingSeconds": None,
                    }
                )
                self._stage2_started_at_monotonic = None
                self._stage2_started_at_ms = None

            self._stage2_progress = next_payload
            self.last_task_message = str(next_payload.get("message", self.last_task_message))
        self.publish_event("stage2-progress", next_payload)

    async def iter_events(self):
        while True:
            event = await self._event_queue.get()
            yield (
                f"event: {event['event']}\n"
                f"data: {json.dumps(event['payload'], ensure_ascii=False, separators=(',', ':'))}\n\n"
            )

    async def watch_active_folder(self) -> None:
        while True:
            await asyncio.sleep(self.watch_interval_seconds)
            if self.active_folder is None:
                continue
            try:
                await asyncio.to_thread(self.refresh_active_folder, True)
            except (DesktopServiceError, Stage2Error) as exc:
                self.last_task_message = str(exc)
                self.publish_event("folder-refresh-failed", {"message": str(exc)})

    def start_watch_loop(self) -> None:
        if self._watch_task is None or self._watch_task.done():
            self._watch_task = asyncio.create_task(self.watch_active_folder())

    def _reset_stage2_progress_for_signature(self, encoder_signature: str) -> None:
        if encoder_signature != "stage1":
            self._stage2_progress = {
                "status": "ready",
                "phase": "finish",
                "current": 1,
                "total": 1,
                "startedAtMs": None,
                "elapsedSeconds": 0,
                "remainingSeconds": 0,
                "message": "Stage 2 adaptation is ready.",
            }
            self._stage2_started_at_monotonic = None
            self._stage2_started_at_ms = None
            return

        self._stage2_progress = {
            "status": "idle",
            "phase": "idle",
            "current": 0,
            "total": 0,
            "startedAtMs": None,
            "elapsedSeconds": 0,
            "remainingSeconds": None,
            "message": "Stage 2 adaptation is idle.",
        }
        self._stage2_started_at_monotonic = None
        self._stage2_started_at_ms = None

    def _require_active_folder(self) -> Path:
        if self.active_folder is None:
            raise DesktopServiceError("Choose a folder before using the desktop app.")
        return self.active_folder

    def _require_indexing_components(self) -> None:
        if self.store is None:
            raise DesktopServiceError("The local index store is not available.")
        if self.encoder is None and self.encoder_loader is None:
            raise DesktopServiceError("The local search encoder is not available.")

    def _resolve_folder_signature(self, folder_path: Path) -> str:
        if self.store is None:
            return "stage1"
        row = self.store.get_folder_state(folder_path.as_posix())
        if row is None:
            return "stage1"
        return str(row["active_encoder_signature"])

    def _load_encoder(self, folder_path: Path | None, encoder_signature: str) -> EncoderProtocol:
        if self.encoder_loader is None:
            if self.encoder is None:
                raise DesktopServiceError("The local search encoder is not available.")
            return self.encoder

        cache_key = self._encoder_cache_key(folder_path, encoder_signature)
        if self.encoder is not None and self._loaded_encoder_key == cache_key:
            return self.encoder

        try:
            next_encoder = self.encoder_loader(folder_path, encoder_signature)
        except Exception as exc:
            raise DesktopServiceError(f"Failed to load the local search encoder: {exc}") from exc

        self.encoder = next_encoder
        self._loaded_encoder_key = cache_key
        return next_encoder

    @staticmethod
    def _encoder_cache_key(folder_path: Path | None, encoder_signature: str) -> tuple[str | None, str]:
        if encoder_signature == "stage1":
            return (None, encoder_signature)
        if folder_path is None:
            return (None, encoder_signature)
        return (folder_path.expanduser().resolve().as_posix(), encoder_signature)

    def _resolve_active_file(self, image_path: str) -> Path:
        folder = self._require_active_folder()
        candidate = (folder / unquote(image_path)).resolve()
        try:
            candidate.relative_to(folder)
        except ValueError as exc:
            raise DesktopServiceError("Image not found.") from exc
        if not candidate.is_file():
            raise DesktopServiceError("Image not found.")
        return candidate

    def _result_payload(self, absolute_path: str) -> dict[str, object]:
        file_path = Path(absolute_path).expanduser().resolve()
        folder = self._require_active_folder()
        relative_path = quote(file_path.relative_to(folder).as_posix(), safe="/")
        return {
            "path": file_path.as_posix(),
            "relativePath": file_path.relative_to(folder).as_posix(),
            "fileName": file_path.name,
            "name": file_path.stem,
            "thumbnailUrl": f"/thumbs/{relative_path}",
            "fullUrl": f"/images/{relative_path}",
            "metadataUrl": f"/api/metadata/{relative_path}",
            "deleteUrl": f"/api/images/{relative_path}",
            "similarUrl": f"/api/similar/{relative_path}",
        }

    @staticmethod
    def _scan_folder(folder_path: Path) -> FolderScanState:
        entries = collect_folder_scan_entries(folder_path)
        rows = tuple((entry.path, entry.relative_path, entry.byte_size, entry.mtime_ns) for entry in entries)
        total_bytes = sum(entry.byte_size for entry in entries)
        return FolderScanState(
            file_count=len(entries),
            total_bytes=total_bytes,
            scan_signature=build_scan_signature(
                [(relative_path, byte_size, mtime_ns) for _, relative_path, byte_size, mtime_ns in rows]
            ),
            rows=rows,
        )

    def _trash_active_files(self, image_paths: list[str]) -> dict[str, object]:
        folder = self._require_active_folder()
        unique_paths: list[Path] = []
        thumbnail_paths: dict[str, Path] = {}
        seen_paths: set[str] = set()
        missing: list[str] = []

        for image_path in image_paths:
            try:
                file_path = self._resolve_active_file(image_path)
            except DesktopServiceError:
                missing.append(unquote(image_path))
                continue
            key = file_path.as_posix()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            unique_paths.append(file_path)
            thumbnail_paths[key] = self._thumbnail_cache_path(file_path)

        if unique_paths:
            trash_manager = self.trash_manager or move_paths_to_trash
            try:
                trash_manager(unique_paths)
            except Exception as exc:
                raise DesktopServiceError(f"Failed to move the selected images to the Trash: {exc}") from exc

            for file_path in unique_paths:
                self._metadata_cache.pop(file_path.as_posix(), None)
                self._delete_thumbnail(thumbnail_paths.get(file_path.as_posix()))

            self._drop_deleted_paths_from_index(unique_paths, folder)

        count = len(unique_paths)
        message = (
            f"Moved {count} image to the Trash."
            if count == 1
            else f"Moved {count} images to the Trash."
        )
        self.last_task_message = message
        deleted = [self._deleted_payload(path, folder) for path in unique_paths]
        return {"deleted": deleted, "missing": missing, "message": message}

    @staticmethod
    def _deleted_payload(file_path: Path, folder: Path) -> dict[str, object]:
        resolved = file_path.expanduser().resolve()
        return {
            "fileName": resolved.name,
            "path": resolved.as_posix(),
            "relativePath": resolved.relative_to(folder).as_posix(),
        }

    def _drop_deleted_paths_from_index(self, deleted_paths: list[Path], folder: Path) -> None:
        removed_absolute_paths = [path.expanduser().resolve().as_posix() for path in deleted_paths]
        self._active_search_view = self._active_search_view.without_paths(removed_absolute_paths)

        if self.store is not None:
            with self.store.transaction():
                for absolute_path in removed_absolute_paths:
                    self.store.mark_path_missing(absolute_path)

                present_rows = [row for row in self.store.get_known_paths(folder.as_posix()) if row["is_present"]]
                signature_rows = [
                    (
                        Path(row["absolute_path"]).relative_to(folder).as_posix(),
                        row["byte_size"],
                        row["mtime_ns"],
                    )
                    for row in present_rows
                ]
                self._active_scan_signature = build_scan_signature(signature_rows)
                self.store.upsert_folder_state(
                    folder.as_posix(),
                    self.active_encoder_signature,
                    file_count=len(signature_rows),
                    total_bytes=sum(row["byte_size"] for row in present_rows),
                    scan_signature=self._active_scan_signature,
                )
        else:
            self._active_scan_signature = ""

        indexed_count = len(self._active_search_view.paths)
        self._index_progress = {
            "status": "ready",
            "phase": "finish",
            "current": indexed_count,
            "total": indexed_count,
            "startedAtMs": None,
            "elapsedSeconds": 0,
            "remainingSeconds": 0,
            "embeddedCount": 0,
            "reusedCount": 0,
            "message": "The folder index is ready.",
        }

    def _read_image_metadata(self, file_path: Path) -> dict[str, object]:
        cache_key = file_path.as_posix()
        cached = self._metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        stat = file_path.stat()
        width = 0
        height = 0
        timestamp_label = "File time"
        timestamp_value = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        try:
            with Image.open(file_path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                exif = image.getexif()
                for exif_tag in (36867, 36868, 306):
                    if exif_tag not in exif:
                        continue
                    formatted = self._format_timestamp(exif.get(exif_tag))
                    if formatted:
                        timestamp_label = "Capture time"
                        timestamp_value = formatted
                        break
        except Exception:
            width = 0
            height = 0

        payload = {
            "path": file_path.as_posix(),
            "relativePath": file_path.relative_to(self._require_active_folder()).as_posix(),
            "fileName": file_path.name,
            "byteSize": stat.st_size,
            "mtimeNs": stat.st_mtime_ns,
            "width": width,
            "height": height,
            "timeLabel": timestamp_label,
            "timeValue": timestamp_value,
        }
        self._metadata_cache[cache_key] = payload
        return payload

    def _encode_uploaded_image(self, file_bytes: bytes, filename: str | None) -> np.ndarray:
        if not file_bytes:
            raise DesktopServiceError("Image is empty.")
        if len(file_bytes) > MAX_IMAGE_UPLOAD_BYTES:
            raise DesktopServiceError("Image is too large.")

        folder = self._require_active_folder()
        encoder = self._load_encoder(folder, self.active_encoder_signature)
        stem = Path(filename or "pasted-image").stem
        safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem).strip("-_")
        if not safe_stem:
            safe_stem = "pasted-image"

        temp_path: Path | None = None
        try:
            with Image.open(io.BytesIO(file_bytes)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                with tempfile.NamedTemporaryFile(
                    prefix=f"semanticgallery-{safe_stem}-",
                    suffix=".jpg",
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                image.save(temp_path, format="JPEG", quality=92)
        except Exception as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise DesktopServiceError("Unsupported image payload.") from exc

        try:
            return encoder.encode_image(temp_path)
        except Exception as exc:
            raise DesktopServiceError(f"Failed to encode the uploaded image: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _format_timestamp(raw_value: object) -> str | None:
        text = str(raw_value).strip()
        if not text:
            return None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        return text

    def _thumbnail_root(self) -> Path:
        if self.thumbnails_dir is not None:
            root = self.thumbnails_dir.expanduser().resolve()
        else:
            root = Path.home() / "Library" / "Caches" / "SemanticGallery" / "thumbs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _thumbnail_cache_path(self, file_path: Path) -> Path:
        stat = file_path.stat()
        cache_key = f"{file_path.as_posix()}:{stat.st_mtime_ns}:{stat.st_size}"
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return self._thumbnail_root() / f"{digest}.jpg"

    def _ensure_thumbnail(self, file_path: Path) -> Path:
        target = self._thumbnail_cache_path(file_path)
        if target.exists():
            return target

        with Image.open(file_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            image.save(target, format="JPEG", quality=88, optimize=True)
        return target

    @staticmethod
    def _delete_thumbnail(thumbnail_path: Path | None) -> None:
        if thumbnail_path is None:
            return
        try:
            thumbnail_path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _render_image_as_jpeg(file_path: Path) -> bytes:
        with Image.open(file_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def _serve_original_image(self, file_path: Path) -> ServedImage:
        if file_path.suffix.lower() in HEIF_SUFFIXES:
            if register_heif_opener is None:
                raise DesktopServiceError("HEIF support is not installed.")
            return ServedImage(media_type="image/jpeg", content=self._render_image_as_jpeg(file_path))

        media_type, _ = mimetypes.guess_type(file_path.name)
        return ServedImage(media_type=media_type or "application/octet-stream", file_path=file_path)
