from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from desktop_runtime.api import build_app
from desktop_runtime.index_store import IndexStore
from desktop_runtime.service import DesktopService
from desktop_runtime.stage2_jobs import Stage2Error


class FakeEncoder:
    def encode_image(self, path: Path) -> np.ndarray:
        if "cat" in path.stem.lower():
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)

    def encode_text(self, query_text: str) -> np.ndarray:
        if "cat" in query_text.lower():
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


class FakeStage2Job:
    def __init__(self, *, result: str = "stage2-signature", error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[Path] = []

    def run(self, folder_path: Path) -> str:
        self.calls.append(folder_path)
        if self.error is not None:
            raise self.error
        return self.result


class BrokenEncoder(FakeEncoder):
    def encode_image(self, path: Path) -> np.ndarray:
        raise RuntimeError(f"cannot encode {path.name}")


class RecordingEncoderLoader:
    def __init__(self):
        self.calls: list[tuple[Path | None, str]] = []

    def __call__(self, folder_path: Path | None, encoder_signature: str) -> FakeEncoder:
        resolved = folder_path.resolve() if folder_path is not None else None
        self.calls.append((resolved, encoder_signature))
        return FakeEncoder()


class RecordingTrashManager:
    def __init__(self, trash_root: Path):
        self.trash_root = trash_root
        self.calls: list[list[Path]] = []

    def __call__(self, paths: list[Path]) -> None:
        resolved_paths = [path.expanduser().resolve() for path in paths]
        self.calls.append(resolved_paths)
        self.trash_root.mkdir(parents=True, exist_ok=True)
        for path in resolved_paths:
            destination = self.trash_root / path.name
            counter = 1
            while destination.exists():
                destination = self.trash_root / f"{path.stem}-{counter}{path.suffix}"
                counter += 1
            path.replace(destination)


class FakeEventService:
    def runtime_status(self) -> dict[str, object]:
        return {
            "setupStatus": "ready",
            "activeFolder": None,
            "activeEncoderSignature": "stage1",
            "lastTaskMessage": "Ready",
            "indexedImageCount": 0,
            "indexing": {
                "status": "idle",
                "phase": "idle",
                "current": 0,
                "total": 0,
                "embeddedCount": 0,
                "reusedCount": 0,
                "message": "Ready",
            },
        }

    async def iter_events(self):
        yield "event: runtime\ndata: {\"message\":\"ready\"}\n\n"


class ThreadedProgressService:
    def __init__(self):
        self._event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self.started = threading.Event()
        self.release = threading.Event()

    def attach_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._event_loop = loop

    def runtime_status(self) -> dict[str, object]:
        return {
            "setupStatus": "ready",
            "activeFolder": None,
            "activeEncoderSignature": "stage1",
            "lastTaskMessage": "Ready",
            "indexedImageCount": 0,
            "indexing": {
                "status": "idle",
                "phase": "idle",
                "current": 0,
                "total": 0,
                "embeddedCount": 0,
                "reusedCount": 0,
                "message": "Ready",
            },
        }

    def set_active_folder(self, folder_path: str) -> dict[str, object]:
        self.started.set()
        if self._event_loop is None:
            raise AssertionError("The event loop should be attached before indexing starts.")
        self._event_loop.call_soon_threadsafe(
            self._event_queue.put_nowait,
            {
                "event": "folder-index-progress",
                "payload": {
                    "status": "running",
                    "phase": "progress",
                    "current": 1,
                    "total": 3,
                    "embeddedCount": 1,
                    "reusedCount": 0,
                    "message": "Indexing cat.jpg (1/3)",
                },
            },
        )
        self.release.wait(timeout=2.0)
        return {
            "setupStatus": "ready",
            "activeFolder": folder_path,
            "activeEncoderSignature": "stage1",
            "lastTaskMessage": "The active folder index is ready.",
            "indexedImageCount": 3,
            "indexing": {
                "status": "ready",
                "phase": "finish",
                "current": 3,
                "total": 3,
                "embeddedCount": 3,
                "reusedCount": 0,
                "message": "The folder index is ready.",
            },
        }

    async def iter_events(self):
        while True:
            event = await self._event_queue.get()
            yield (
                f"event: {event['event']}\n"
                f"data: {json.dumps(event['payload'], separators=(',', ':'))}\n\n"
            )


class DesktopApiTests(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path, *, size: tuple[int, int] = (720, 480), color: tuple[int, int, int] = (200, 90, 60)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path, format="JPEG")

    def _build_service(
        self,
        root: Path,
        *,
        stage2_job: FakeStage2Job | None = None,
        encoder_loader: RecordingEncoderLoader | None = None,
        trash_manager: RecordingTrashManager | None = None,
    ) -> DesktopService:
        store = IndexStore.connect(root / "index.sqlite3")
        store.migrate()
        return DesktopService(
            store=store,
            encoder=None if encoder_loader is not None else FakeEncoder(),
            encoder_loader=encoder_loader,
            stage2_job=stage2_job or FakeStage2Job(),
            thumbnails_dir=root / "thumbs",
            trash_manager=trash_manager,
        )

    def test_runtime_status_endpoint_returns_setup_state(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            service = self._build_service(root)
            service.setup_status = "ready"
            client = TestClient(build_app(service))

            response = client.get("/api/runtime/status")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["setupStatus"], "ready")
            self.assertEqual(response.json()["activeEncoderSignature"], "stage1")

    def test_activate_folder_sets_the_active_folder_without_reindexing(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            client = TestClient(build_app(self._build_service(root)))

            response = client.post(
                "/api/folders/activate",
                json={"folderPath": folder.as_posix(), "encoderSignature": "stage1"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["activeFolder"], folder.resolve().as_posix())
            self.assertEqual(payload["activeEncoderSignature"], "stage1")

    def test_encode_text_endpoint_returns_a_vector_for_the_selected_signature(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            client = TestClient(build_app(self._build_service(root)))

            response = client.post(
                "/api/encode/text",
                json={
                    "folderPath": folder.as_posix(),
                    "encoderSignature": "stage1",
                    "queryText": "cat",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["vector"], [1.0, 0.0])

    def test_encode_image_path_endpoint_returns_a_vector_for_the_selected_signature(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            image_path = folder / "cat.jpg"
            self._write_image(image_path)
            client = TestClient(build_app(self._build_service(root)))

            response = client.post(
                "/api/encode/image-path",
                json={
                    "folderPath": folder.as_posix(),
                    "encoderSignature": "stage1",
                    "imagePath": image_path.as_posix(),
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["vector"], [1.0, 0.0])

    def test_runtime_status_preflight_allows_tauri_webview_requests(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            client = TestClient(build_app(self._build_service(root)))

            response = client.options(
                "/api/runtime/status",
                headers={
                    "Origin": "tauri://localhost",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "content-type",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"], "*")
            self.assertIn("GET", response.headers["access-control-allow-methods"])

    def test_select_folder_builds_searchable_view_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            cat = folder / "cat.jpg"
            dog = folder / "dog.jpg"
            self._write_image(cat, size=(640, 480), color=(220, 120, 90))
            self._write_image(dog, size=(320, 200), color=(90, 120, 220))

            client = TestClient(build_app(self._build_service(root)))

            select_response = client.post("/api/folders/select", json={"folderPath": folder.as_posix()})
            search_response = client.get("/api/search", params={"q": "cat", "limit": 5})
            metadata_response = client.get("/api/metadata/cat.jpg")

            self.assertEqual(select_response.status_code, 200)
            self.assertEqual(select_response.json()["activeFolder"], folder.resolve().as_posix())
            self.assertEqual(search_response.status_code, 200)
            self.assertEqual(search_response.json()["results"][0]["path"], cat.resolve().as_posix())
            self.assertEqual(metadata_response.status_code, 200)
            self.assertEqual(metadata_response.json()["fileName"], "cat.jpg")
            self.assertEqual(metadata_response.json()["width"], 640)
            self.assertEqual(metadata_response.json()["height"], 480)
            self.assertIn(metadata_response.json()["timeLabel"], {"File time", "Capture time"})
            self.assertTrue(metadata_response.json()["timeValue"])

    def test_search_results_include_media_urls_and_similar_search(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            self._write_image(folder / "cat.jpg", color=(220, 120, 90))
            self._write_image(folder / "cat-friend.jpg", color=(210, 100, 80))
            self._write_image(folder / "dog.jpg", color=(90, 120, 220))

            client = TestClient(build_app(self._build_service(root)))
            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})

            search_response = client.get("/api/search", params={"q": "cat", "limit": 3})
            similar_response = client.get("/api/similar/cat.jpg", params={"limit": 2})

            self.assertEqual(search_response.status_code, 200)
            first = search_response.json()["results"][0]
            self.assertTrue(first["thumbnailUrl"].startswith("/thumbs/"))
            self.assertTrue(first["fullUrl"].startswith("/images/"))
            self.assertTrue(first["metadataUrl"].startswith("/api/metadata/"))
            self.assertTrue(first["deleteUrl"].startswith("/api/images/"))
            self.assertTrue(first["similarUrl"].startswith("/api/similar/"))
            self.assertTrue({"cat.jpg", "cat-friend.jpg"}.issubset({row["relativePath"] for row in search_response.json()["results"]}))

            self.assertEqual(similar_response.status_code, 200)
            self.assertEqual(similar_response.json()["results"][0]["relativePath"], "cat-friend.jpg")
            self.assertNotEqual(similar_response.json()["results"][0]["relativePath"], "cat.jpg")

    def test_image_search_endpoint_accepts_uploaded_queries(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            self._write_image(folder / "cat.jpg", color=(220, 120, 90))
            self._write_image(folder / "dog.jpg", color=(90, 120, 220))

            client = TestClient(build_app(self._build_service(root)))
            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})

            image_buffer = io.BytesIO()
            Image.new("RGB", (640, 480), (220, 120, 90)).save(image_buffer, format="PNG")
            image_buffer.seek(0)

            response = client.post(
                "/api/search/image",
                params={"limit": 3},
                files={"image": ("cat-query.png", image_buffer, "image/png")},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["results"][0]["relativePath"], "cat.jpg")

    def test_thumbnail_and_full_image_endpoints_return_image_bytes(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            cat = folder / "cat.jpg"
            self._write_image(cat, size=(1024, 640), color=(220, 120, 90))

            client = TestClient(build_app(self._build_service(root)))
            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})

            thumbnail_response = client.get("/thumbs/cat.jpg")
            image_response = client.get("/images/cat.jpg")

            self.assertEqual(thumbnail_response.status_code, 200)
            self.assertEqual(thumbnail_response.headers["content-type"], "image/jpeg")
            self.assertGreater(len(thumbnail_response.content), 0)

            self.assertEqual(image_response.status_code, 200)
            self.assertEqual(image_response.headers["content-type"], "image/jpeg")
            self.assertGreater(len(image_response.content), 0)

    def test_delete_image_moves_the_file_to_trash_and_refreshes_the_index(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            cat = folder / "cat.jpg"
            cat_friend = folder / "cat-friend.jpg"
            dog = folder / "dog.jpg"
            self._write_image(cat, color=(220, 120, 90))
            self._write_image(cat_friend, color=(210, 100, 80))
            self._write_image(dog, color=(90, 120, 220))

            trash_manager = RecordingTrashManager(root / "trash")
            client = TestClient(build_app(self._build_service(root, trash_manager=trash_manager)))
            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})

            delete_response = client.delete("/api/images/cat.jpg")
            search_response = client.get("/api/search", params={"q": "cat", "limit": 5})
            status_response = client.get("/api/runtime/status")

            self.assertEqual(delete_response.status_code, 200)
            self.assertEqual(delete_response.json()["fileName"], "cat.jpg")
            self.assertIn("Trash", delete_response.json()["message"])
            self.assertFalse(cat.exists())
            self.assertEqual(trash_manager.calls, [[cat.resolve()]])
            self.assertEqual(status_response.json()["indexedImageCount"], 2)
            returned_paths = {row["relativePath"] for row in search_response.json()["results"]}
            self.assertNotIn("cat.jpg", returned_paths)

    def test_delete_image_updates_index_without_triggering_full_refresh(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            cat = folder / "cat.jpg"
            dog = folder / "dog.jpg"
            self._write_image(cat, color=(220, 120, 90))
            self._write_image(dog, color=(90, 120, 220))

            trash_manager = RecordingTrashManager(root / "trash")
            service = self._build_service(root, trash_manager=trash_manager)
            client = TestClient(build_app(service))
            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})

            def fail_refresh(*_args, **_kwargs):
                raise AssertionError("delete should not trigger a full folder refresh")

            service.refresh_active_folder = fail_refresh  # type: ignore[method-assign]

            delete_response = client.delete("/api/images/cat.jpg")
            status_response = client.get("/api/runtime/status")
            search_response = client.get("/api/search", params={"q": "cat", "limit": 5})

            self.assertEqual(delete_response.status_code, 200)
            self.assertEqual(status_response.json()["indexedImageCount"], 1)
            self.assertFalse(cat.exists())
            self.assertEqual(trash_manager.calls, [[cat.resolve()]])
            returned_paths = {row["relativePath"] for row in search_response.json()["results"]}
            self.assertNotIn("cat.jpg", returned_paths)

    def test_batch_delete_moves_only_selected_files_to_trash(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            cat = folder / "cat.jpg"
            cat_friend = folder / "cat-friend.jpg"
            dog = folder / "dog.jpg"
            self._write_image(cat, color=(220, 120, 90))
            self._write_image(cat_friend, color=(210, 100, 80))
            self._write_image(dog, color=(90, 120, 220))

            trash_manager = RecordingTrashManager(root / "trash")
            client = TestClient(build_app(self._build_service(root, trash_manager=trash_manager)))
            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})

            delete_response = client.post(
                "/api/images/batch-delete",
                json={"paths": ["cat.jpg", "cat-friend.jpg", "cat.jpg", "missing.jpg"]},
            )
            status_response = client.get("/api/runtime/status")

            self.assertEqual(delete_response.status_code, 200)
            self.assertEqual([item["relativePath"] for item in delete_response.json()["deleted"]], ["cat.jpg", "cat-friend.jpg"])
            self.assertEqual(delete_response.json()["missing"], ["missing.jpg"])
            self.assertIn("Moved 2 images to the Trash.", delete_response.json()["message"])
            self.assertEqual(trash_manager.calls, [[cat.resolve(), cat_friend.resolve()]])
            self.assertFalse(cat.exists())
            self.assertFalse(cat_friend.exists())
            self.assertTrue(dog.exists())
            self.assertEqual(status_response.json()["indexedImageCount"], 1)

    def test_batch_delete_updates_index_without_triggering_full_refresh(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            cat = folder / "cat.jpg"
            cat_friend = folder / "cat-friend.jpg"
            dog = folder / "dog.jpg"
            self._write_image(cat, color=(220, 120, 90))
            self._write_image(cat_friend, color=(210, 100, 80))
            self._write_image(dog, color=(90, 120, 220))

            trash_manager = RecordingTrashManager(root / "trash")
            service = self._build_service(root, trash_manager=trash_manager)
            client = TestClient(build_app(service))
            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})

            def fail_refresh(*_args, **_kwargs):
                raise AssertionError("batch delete should not trigger a full folder refresh")

            service.refresh_active_folder = fail_refresh  # type: ignore[method-assign]

            delete_response = client.post(
                "/api/images/batch-delete",
                json={"paths": ["cat.jpg", "cat-friend.jpg"]},
            )
            status_response = client.get("/api/runtime/status")

            self.assertEqual(delete_response.status_code, 200)
            self.assertEqual(status_response.json()["indexedImageCount"], 1)
            self.assertEqual(trash_manager.calls, [[cat.resolve(), cat_friend.resolve()]])
            self.assertFalse(cat.exists())
            self.assertFalse(cat_friend.exists())
            self.assertTrue(dog.exists())

    def test_refresh_active_folder_skips_when_lightweight_scan_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            (folder / "cat.jpg").write_bytes(b"cat")
            (folder / "dog.jpg").write_bytes(b"dog")

            service = self._build_service(root)
            service.set_active_folder(folder.as_posix())

            skipped = service.refresh_active_folder(lightweight=True)
            (folder / "bird.jpg").write_bytes(b"bird")
            refreshed = service.refresh_active_folder(lightweight=True)

            self.assertTrue(skipped["skipped"])
            self.assertFalse(refreshed["skipped"])
            self.assertEqual(refreshed["indexedImageCount"], 3)

    def test_select_folder_failure_restores_previous_active_folder(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            good_folder = root / "good-gallery"
            bad_folder = root / "bad-gallery"
            good_folder.mkdir()
            bad_folder.mkdir()
            (good_folder / "cat.jpg").write_bytes(b"cat")
            (bad_folder / "dog.jpg").write_bytes(b"dog")

            store = IndexStore.connect(root / "index.sqlite3")
            store.migrate()
            service = DesktopService(store=store, encoder=BrokenEncoder(), stage2_job=FakeStage2Job())
            service.active_folder = good_folder.resolve()

            client = TestClient(build_app(service))
            response = client.post("/api/folders/select", json={"folderPath": bad_folder.as_posix()})

            self.assertEqual(response.status_code, 400)
            self.assertEqual(service.active_folder, good_folder.resolve())
            self.assertIn("Failed to index the selected folder", response.json()["detail"])

    def test_stage2_endpoint_returns_readable_error(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            (folder / "cat.jpg").write_bytes(b"cat")

            stage2_job = FakeStage2Job(
                error=Stage2Error(
                    "This folder currently has 1 supported images. Stage 2 adaptation is only useful for folders with at least 100 images, so the app will skip adaptation for now."
                )
            )
            service = self._build_service(root, stage2_job=stage2_job)
            service.active_folder = folder.resolve()
            client = TestClient(build_app(service))

            response = client.post("/api/stage2/run")

            self.assertEqual(response.status_code, 400)
            self.assertIn("at least 100 images", response.json()["detail"])

    def test_stage2_endpoint_switches_active_encoder_signature_after_success(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            for index in range(100):
                (folder / f"cat-{index}.jpg").write_bytes(b"cat")

            stage2_job = FakeStage2Job(result="stage2-signature")
            service = self._build_service(root, stage2_job=stage2_job)
            client = TestClient(build_app(service))

            client.post("/api/folders/select", json={"folderPath": folder.as_posix()})
            response = client.post("/api/stage2/run")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["activeEncoderSignature"], "stage2-signature")
            self.assertEqual(stage2_job.calls, [folder.resolve()])

    def test_folder_selection_restores_folder_specific_encoder_signature_after_stage2(self):
        with tempfile.TemporaryDirectory(prefix="sg-desktop-api-") as tmp_dir:
            root = Path(tmp_dir)
            first_folder = root / "gallery-one"
            second_folder = root / "gallery-two"
            first_folder.mkdir()
            second_folder.mkdir()
            for index in range(100):
                (first_folder / f"cat-{index}.jpg").write_bytes(f"cat-{index}".encode("utf-8"))
            (second_folder / "dog.jpg").write_bytes(b"dog")

            encoder_loader = RecordingEncoderLoader()
            service = self._build_service(
                root,
                stage2_job=FakeStage2Job(result="stage2-gallery-one"),
                encoder_loader=encoder_loader,
            )
            client = TestClient(build_app(service))

            first_select = client.post("/api/folders/select", json={"folderPath": first_folder.as_posix()})
            stage2_response = client.post("/api/stage2/run")
            second_select = client.post("/api/folders/select", json={"folderPath": second_folder.as_posix()})
            restored_select = client.post("/api/folders/select", json={"folderPath": first_folder.as_posix()})

            self.assertEqual(first_select.status_code, 200)
            self.assertEqual(stage2_response.status_code, 200)
            self.assertEqual(second_select.status_code, 200)
            self.assertEqual(second_select.json()["activeEncoderSignature"], "stage1")
            self.assertEqual(restored_select.status_code, 200)
            self.assertEqual(restored_select.json()["activeEncoderSignature"], "stage2-gallery-one")
            self.assertEqual(encoder_loader.calls[-1], (first_folder.resolve(), "stage2-gallery-one"))

    def test_events_endpoint_streams_server_sent_events(self):
        client = TestClient(build_app(FakeEventService()))

        with client.stream("GET", "/api/events") as response:
            lines = list(response.iter_lines())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: runtime", lines)
        self.assertIn('data: {"message":"ready"}', lines)


class DesktopApiAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_select_folder_progress_reaches_consumers_before_response_finishes(self):
        service = ThreadedProgressService()
        app = build_app(service)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            event_task = asyncio.create_task(service.iter_events().__anext__())
            select_task = asyncio.create_task(client.post("/api/folders/select", json={"folderPath": "/tmp/gallery"}))

            started = await asyncio.to_thread(service.started.wait, 1.0)
            self.assertTrue(started)

            progress = await asyncio.wait_for(event_task, timeout=2.0)
            self.assertIn("Indexing cat.jpg (1/3)", progress)
            self.assertFalse(select_task.done())

            service.release.set()
            response = await asyncio.wait_for(select_task, timeout=2.0)

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["activeFolder"], "/tmp/gallery")


class DesktopServiceEventLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_event_from_worker_thread_reaches_event_stream(self):
        service = DesktopService()
        service.attach_event_loop(asyncio.get_running_loop())

        events = service.iter_events()
        event_task = asyncio.create_task(events.__anext__())

        await asyncio.to_thread(
            service.publish_event,
            "folder-index-progress",
            {
                "status": "running",
                "phase": "progress",
                "current": 1,
                "total": 2,
                "embeddedCount": 1,
                "reusedCount": 0,
                "message": "Indexing sample.jpg (1/2)",
            },
        )
        payload = await asyncio.wait_for(event_task, timeout=2.0)

        self.assertIn("event: folder-index-progress", payload)
        self.assertIn('"message":"Indexing sample.jpg (1/2)"', payload)


if __name__ == "__main__":
    unittest.main()
