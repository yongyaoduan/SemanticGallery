from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from desktop_runtime.index_store import IndexStore
from desktop_runtime.service import DesktopService


class FakeEncoder:
    def encode_image(self, path: Path) -> np.ndarray:
        return np.asarray([float(len(path.name)), 1.0], dtype=np.float32)

    def encode_text(self, query_text: str) -> np.ndarray:
        return np.asarray([float(len(query_text)), 1.0], dtype=np.float32)


class DesktopServiceTests(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 240), (120, 90, 200)).save(path, format="JPEG")

    def _build_service(self, root: Path) -> DesktopService:
        store = IndexStore.connect(root / "index.sqlite3")
        store.migrate()
        return DesktopService(
            store=store,
            encoder=FakeEncoder(),
            thumbnails_dir=root / "thumbs",
            setup_status="ready",
        )

    def test_runtime_status_remains_available_while_refresh_runs(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-service-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            self._write_image(folder / "cat.jpg")

            service = self._build_service(root)
            service.set_active_folder(folder.as_posix())

            refresh_started = threading.Event()
            release_refresh = threading.Event()

            def blocking_reconcile(*args, **kwargs):
                del args, kwargs
                refresh_started.set()
                release_refresh.wait(timeout=2.0)

            with patch("desktop_runtime.service.reconcile_folder", side_effect=blocking_reconcile):
                refresh_thread = threading.Thread(
                    target=lambda: service.refresh_active_folder(lightweight=False),
                    daemon=True,
                )
                refresh_thread.start()
                self.assertTrue(refresh_started.wait(timeout=1.0))

                status_holder: dict[str, object] = {}
                status_thread = threading.Thread(
                    target=lambda: status_holder.setdefault("value", service.runtime_status()),
                    daemon=True,
                )
                status_thread.start()
                status_thread.join(timeout=0.2)

                self.assertFalse(status_thread.is_alive(), "runtime_status should stay readable during refresh")
                self.assertEqual(status_holder["value"]["activeFolder"], folder.resolve().as_posix())

                release_refresh.set()
                refresh_thread.join(timeout=1.0)
                self.assertFalse(refresh_thread.is_alive(), "refresh thread should finish after reconciliation is released")

    def test_index_progress_tracks_elapsed_and_remaining_time(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-service-") as tmp_dir:
            root = Path(tmp_dir)
            service = self._build_service(root)

            with patch("desktop_runtime.service.time.time", return_value=1_700_000_000), patch(
                "desktop_runtime.service.time.monotonic",
                side_effect=[10.0, 10.0, 25.0, 34.0],
            ):
                service._publish_index_progress(
                    {
                        "status": "running",
                        "phase": "start",
                        "current": 0,
                        "total": 4,
                        "message": "Scanning the selected folder for index updates.",
                    }
                )
                service._publish_index_progress(
                    {
                        "status": "running",
                        "phase": "progress",
                        "current": 2,
                        "total": 4,
                        "message": "Indexing cat.jpg (2/4)",
                    }
                )
                service._publish_index_progress(
                    {
                        "status": "ready",
                        "phase": "finish",
                        "current": 4,
                        "total": 4,
                        "message": "The folder index is ready.",
                    }
                )

            self.assertEqual(service.runtime_status()["indexing"]["startedAtMs"], 1_700_000_000_000)
            self.assertEqual(service.runtime_status()["indexing"]["elapsedSeconds"], 24)
            self.assertEqual(service.runtime_status()["indexing"]["remainingSeconds"], 0)


if __name__ == "__main__":
    unittest.main()
