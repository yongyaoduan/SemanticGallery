from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

import numpy as np

from deployment.search_utils import is_searchable_query
from desktop_runtime.folder_sync import build_scan_signature, iter_visible_images, reconcile_folder
from desktop_runtime.search_view import ActiveSearchView
from desktop_runtime.stage2_jobs import Stage2Error


class DesktopServiceError(RuntimeError):
    pass


class EncoderProtocol(Protocol):
    def encode_image(self, path: Path) -> np.ndarray: ...

    def encode_text(self, query_text: str) -> np.ndarray: ...


class Stage2JobProtocol(Protocol):
    def run(self, folder_path: Path) -> str: ...


@dataclass(frozen=True)
class FolderScanState:
    file_count: int
    total_bytes: int
    scan_signature: str


@dataclass
class DesktopService:
    store: Any | None = None
    encoder: EncoderProtocol | None = None
    stage2_job: Stage2JobProtocol | None = None
    setup_status: str = "idle"
    active_folder: Path | None = None
    active_encoder_signature: str = "stage1"
    last_task_message: str = ""
    watch_interval_seconds: float = 5.0
    _event_queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue, init=False)
    _watch_task: asyncio.Task[None] | None = field(default=None, init=False)
    _active_scan_signature: str = field(default="", init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _active_search_view: ActiveSearchView = field(
        default_factory=lambda: ActiveSearchView(paths=[], matrix=np.zeros((0, 0), dtype=np.float32)),
        init=False,
    )

    def runtime_status(self) -> dict[str, object]:
        with self._lock:
            return {
                "setupStatus": self.setup_status,
                "activeFolder": self.active_folder.as_posix() if self.active_folder else None,
                "activeEncoderSignature": self.active_encoder_signature,
                "lastTaskMessage": self.last_task_message,
                "indexedImageCount": len(self._active_search_view.paths),
            }

    def set_active_folder(self, folder_path: str) -> dict[str, object]:
        with self._lock:
            resolved = Path(folder_path).expanduser().resolve()
            if not resolved.is_dir():
                raise DesktopServiceError("The selected folder is not available.")

            previous_folder = self.active_folder
            previous_signature = self.active_encoder_signature
            previous_scan_signature = self._active_scan_signature
            previous_view = self._active_search_view

            self.active_folder = resolved
            try:
                return self.refresh_active_folder(lightweight=False)
            except Stage2Error:
                self.active_folder = previous_folder
                self.active_encoder_signature = previous_signature
                self._active_scan_signature = previous_scan_signature
                self._active_search_view = previous_view
                raise
            except DesktopServiceError as exc:
                self.active_folder = previous_folder
                self.active_encoder_signature = previous_signature
                self._active_scan_signature = previous_scan_signature
                self._active_search_view = previous_view
                self.last_task_message = str(exc)
                raise DesktopServiceError(f"Failed to index the selected folder: {exc}") from exc
            except Exception as exc:
                self.active_folder = previous_folder
                self.active_encoder_signature = previous_signature
                self._active_scan_signature = previous_scan_signature
                self._active_search_view = previous_view
                self.last_task_message = str(exc)
                raise DesktopServiceError(f"Failed to index the selected folder: {exc}") from exc

    def refresh_active_folder(self, lightweight: bool = False) -> dict[str, object]:
        with self._lock:
            folder = self._require_active_folder()
            scan_state = self._scan_folder(folder)
            if lightweight and scan_state.scan_signature == self._active_scan_signature:
                self.last_task_message = "The active folder is already up to date."
                return {"refreshed": False, "skipped": True, **self.runtime_status()}

            self._require_indexing_components()
            try:
                reconcile_folder(self.store, folder, self.active_encoder_signature, self.encoder)
                next_view = ActiveSearchView.from_store(self.store, folder.as_posix(), self.active_encoder_signature)
            except (DesktopServiceError, Stage2Error):
                raise
            except Exception as exc:
                self.last_task_message = str(exc)
                raise DesktopServiceError(f"Failed to refresh the active folder: {exc}") from exc

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
        with self._lock:
            folder = self._require_active_folder()
            if self.stage2_job is None:
                raise DesktopServiceError("Stage 2 adaptation is not available.")
            self._require_indexing_components()

            try:
                next_signature = self.stage2_job.run(folder)
                reconcile_folder(self.store, folder, next_signature, self.encoder)
                next_view = ActiveSearchView.from_store(self.store, folder.as_posix(), next_signature)
            except (DesktopServiceError, Stage2Error):
                raise
            except Exception as exc:
                self.last_task_message = str(exc)
                raise DesktopServiceError(f"Failed to finish Stage 2 adaptation: {exc}") from exc

            self.active_encoder_signature = next_signature
            self._active_search_view = next_view
            self._active_scan_signature = self._scan_folder(folder).scan_signature
            self.last_task_message = "Stage 2 adaptation is ready."
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
            if self.encoder is None:
                raise DesktopServiceError("The search encoder is not available.")

            query_vector = self.encoder.encode_text(text)
            matches = self._active_search_view.search(query_vector, limit)
            return {"query": text, "results": [self._result_payload(path) for path in matches]}

    def metadata(self, image_path: str) -> dict[str, object]:
        with self._lock:
            file_path = self._resolve_active_file(image_path)
            stat = file_path.stat()
            return {
                "path": file_path.as_posix(),
                "relativePath": file_path.relative_to(self._require_active_folder()).as_posix(),
                "fileName": file_path.name,
                "byteSize": stat.st_size,
                "mtimeNs": stat.st_mtime_ns,
            }

    def publish_event(self, event_name: str, payload: dict[str, object]) -> None:
        self._event_queue.put_nowait({"event": event_name, "payload": payload})

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
                self.refresh_active_folder(lightweight=True)
            except (DesktopServiceError, Stage2Error) as exc:
                self.last_task_message = str(exc)
                self.publish_event("folder-refresh-failed", {"message": str(exc)})

    def start_watch_loop(self) -> None:
        if self._watch_task is None or self._watch_task.done():
            self._watch_task = asyncio.create_task(self.watch_active_folder())

    def _require_active_folder(self) -> Path:
        if self.active_folder is None:
            raise DesktopServiceError("Choose a folder before using the desktop app.")
        return self.active_folder

    def _require_indexing_components(self) -> None:
        if self.store is None:
            raise DesktopServiceError("The local index store is not available.")
        if self.encoder is None:
            raise DesktopServiceError("The local search encoder is not available.")

    def _resolve_active_file(self, image_path: str) -> Path:
        folder = self._require_active_folder()
        candidate = (folder / image_path).resolve()
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
        return {
            "path": file_path.as_posix(),
            "relativePath": file_path.relative_to(folder).as_posix(),
            "fileName": file_path.name,
            "name": file_path.stem,
        }

    @staticmethod
    def _scan_folder(folder_path: Path) -> FolderScanState:
        rows: list[tuple[str, int, int]] = []
        total_bytes = 0
        for path in iter_visible_images(folder_path):
            stat = path.stat()
            rows.append((path.relative_to(folder_path).as_posix(), stat.st_size, stat.st_mtime_ns))
            total_bytes += stat.st_size
        return FolderScanState(
            file_count=len(rows),
            total_bytes=total_bytes,
            scan_signature=build_scan_signature(rows),
        )
