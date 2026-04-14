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


class FakeStage2Job:
    def __init__(self, signature: str = "stage2-signature"):
        self.signature = signature

    def run(self, folder_path: Path) -> str:
        del folder_path
        return self.signature


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

    @staticmethod
    def _fake_system_thumbnail(_file_path: Path, target: Path) -> bool:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"thumb")
        return True

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

    def test_stage2_progress_tracks_elapsed_and_remaining_time(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-service-") as tmp_dir:
            root = Path(tmp_dir)
            service = self._build_service(root)

            with patch("desktop_runtime.service.time.time", return_value=1_700_000_000), patch(
                "desktop_runtime.service.time.monotonic",
                side_effect=[20.0, 20.0, 32.0, 41.0],
            ):
                service._publish_stage2_progress(
                    {
                        "status": "running",
                        "phase": "prepare",
                        "current": 0,
                        "total": 0,
                        "message": "Preparing private adaptation data.",
                    }
                )
                service._publish_stage2_progress(
                    {
                        "status": "running",
                        "phase": "adapt",
                        "current": 6,
                        "total": 12,
                        "message": "Training epoch 1 of 1 · step 4 of 10",
                    }
                )
                service._publish_stage2_progress(
                    {
                        "status": "ready",
                        "phase": "finish",
                        "current": 12,
                        "total": 12,
                        "message": "Stage 2 adaptation is ready.",
                    }
                )

            self.assertEqual(service.runtime_status()["stage2"]["startedAtMs"], 1_700_000_000_000)
            self.assertEqual(service.runtime_status()["stage2"]["elapsedSeconds"], 21)
            self.assertEqual(service.runtime_status()["stage2"]["remainingSeconds"], 0)

    def test_search_text_prewarms_result_thumbnails(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-service-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            image_path = folder / "cat.jpg"
            self._write_image(image_path)

            service = self._build_service(root)
            service.set_active_folder(folder.as_posix())

            thumbnail_path = service._thumbnail_cache_path(image_path)
            self.assertFalse(thumbnail_path.exists())

            with patch.object(
                service,
                "_generate_system_thumbnail",
                side_effect=self._fake_system_thumbnail,
            ) as generate_thumbnail:
                payload = service.search_text("cat", limit=5)

            self.assertEqual(len(payload["results"]), 1)
            self.assertTrue(thumbnail_path.is_file())
            self.assertEqual(generate_thumbnail.call_count, 1)

    def test_search_text_only_blocks_on_the_first_row_of_result_thumbnails(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-service-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            for index in range(6):
                self._write_image(folder / f"cat-{index}.jpg")

            service = self._build_service(root)
            service.set_active_folder(folder.as_posix())

            synchronous_batches: list[list[str]] = []
            queued_batches: list[list[str]] = []

            with patch.object(
                service,
                "_prewarm_thumbnails",
                side_effect=lambda paths: synchronous_batches.append([path.name for path in paths]),
            ), patch.object(
                service,
                "_schedule_thumbnail_prewarm",
                side_effect=lambda paths: queued_batches.append([path.name for path in paths]),
            ):
                payload = service.search_text("cat", limit=6)

            self.assertEqual(len(payload["results"]), 6)
            self.assertEqual(len(synchronous_batches), 1)
            self.assertEqual(len(synchronous_batches[0]), 5)
            self.assertEqual(len(queued_batches), 1)
            self.assertEqual(len(queued_batches[0]), 1)

    def test_stage2_rebuild_progress_is_published_during_reindex(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-service-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            self._write_image(folder / "cat.jpg")

            service = self._build_service(root)
            service.stage2_job = FakeStage2Job()
            service.set_active_folder(folder.as_posix())

            published: list[dict[str, object]] = []

            def fake_reconcile(*args, **kwargs):
                del args
                emit_progress = kwargs.get("emit_progress")
                if emit_progress is None:
                    return
                emit_progress(
                    {
                        "phase": "start",
                        "current": 0,
                        "total": 4,
                        "embeddedCount": 0,
                        "reusedCount": 0,
                        "message": "Scanning the selected folder for index updates.",
                    }
                )
                emit_progress(
                    {
                        "phase": "progress",
                        "current": 2,
                        "total": 4,
                        "embeddedCount": 1,
                        "reusedCount": 1,
                        "message": "Indexing cat.jpg (2/4)",
                    }
                )
                emit_progress(
                    {
                        "phase": "finish",
                        "current": 4,
                        "total": 4,
                        "embeddedCount": 1,
                        "reusedCount": 1,
                        "message": "The folder index is ready.",
                    }
                )

            with patch.object(service, "_publish_stage2_progress", wraps=service._publish_stage2_progress) as publish_mock:
                with patch("desktop_runtime.service.reconcile_folder", side_effect=fake_reconcile):
                    service.run_stage2_for_active_folder()

            published = [call.args[0] for call in publish_mock.call_args_list]
            self.assertIn("reindex", [payload["phase"] for payload in published])
            running_reindex = [payload for payload in published if payload["phase"] == "reindex" and payload["status"] == "running"]
            self.assertTrue(running_reindex)
            self.assertIn(
                (2, 4),
                [(payload["phaseCurrent"], payload["phaseTotal"]) for payload in running_reindex],
            )
            self.assertEqual(running_reindex[-1]["phaseCurrent"], 4)
            self.assertEqual(running_reindex[-1]["phaseTotal"], 4)


if __name__ == "__main__":
    unittest.main()
